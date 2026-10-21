#!/usr/bin/env Rscript

# Recreate the clustered K=6 ADMIXTURE figure with ggplot2.

suppressPackageStartupMessages(library(ggplot2))

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1L) {
  stop("Could not determine the script directory")
}
script_dir <- dirname(normalizePath(sub("^--file=", "", script_arg)))
repo_dir <- normalizePath(file.path(script_dir, "..", ".."))
source_dir <- file.path(repo_dir, "scripts", "02_population_structure", "source_data")
figure_dir <- file.path(script_dir, "output")
input_path <- file.path(source_dir, "admixture_k6_hierarchical_cluster_order.tsv")
output_stem <- file.path(figure_dir, "admixture_best_k6_hierarchical_clustered")

# First six colors from ggthemes' Miller Stone palette.
ancestry_colors <- c(
  S1 = "#4F6980",
  S2 = "#849DB1",
  S3 = "#A2CEAA",
  S4 = "#638B66",
  S5 = "#BFBB60",
  S6 = "#F47942"
)
region_levels <- c(
  "Western Mediterranean",
  "Central Mediterranean–Adriatic",
  "Eastern Mediterranean–Levant",
  "Iranian Plateau"
)
region_labels <- c(
  "Western Mediterranean" = "West. Med.",
  "Central Mediterranean–Adriatic" = "Central Med.",
  "Eastern Mediterranean–Levant" = "East. Med.",
  "Iranian Plateau" = "Iran"
)
region_colors <- c(
  "Western Mediterranean" = "#6388B4",
  "Central Mediterranean–Adriatic" = "#FFAE34",
  "Eastern Mediterranean–Levant" = "#55AD89",
  "Iranian Plateau" = "#BB7693"
)
subspecies_levels <- c("Europaea", "Sylvestris")
region_gap <- 3

data <- read.delim(
  input_path,
  check.names = FALSE,
  stringsAsFactors = FALSE,
  fileEncoding = "UTF-8"
)
ancestry_columns <- grep("^ancestry_", names(data), value = TRUE)
if (nrow(data) != 212L || anyDuplicated(data$sample_id)) {
  stop("Expected 212 unique samples")
}
if (length(ancestry_columns) != 6L) {
  stop("Expected six ancestry components")
}
if (any(abs(rowSums(data[ancestry_columns]) - 1) > 1e-4)) {
  stop("Ancestry proportions do not sum to one")
}
if (any(!data$geographic_region %in% region_levels)) {
  stop("An unknown geographic region is present")
}

data$subspecies_label <- ifelse(
  data$subspecies == "europaea", "Europaea", "Sylvestris"
)
data$subspecies_label <- factor(data$subspecies_label, levels = subspecies_levels)

# Preserve the existing hierarchical order, but repack samples independently
# within each subspecies facet and retain a visual gap between regions.
data$facet_position <- NA_real_
for (subspecies_name in subspecies_levels) {
  offset <- 0
  for (region_name in region_levels) {
    indices <- which(
      data$subspecies_label == subspecies_name &
        data$geographic_region == region_name
    )
    if (length(indices) == 0L) {
      next
    }
    data$facet_position[indices] <- offset + seq_along(indices) - 1
    offset <- offset + length(indices) + region_gap
  }
}
if (anyNA(data$facet_position)) {
  stop("At least one sample was not assigned a facet position")
}

# Explicit rectangle coordinates preserve the plotted component order:
# ancestry_3, ancestry_1, ancestry_4, ancestry_2, ancestry_5, ancestry_6.
ancestry_rectangles <- vector("list", length(ancestry_columns))
bottom <- rep(0, nrow(data))
for (index in seq_along(ancestry_columns)) {
  values <- data[[ancestry_columns[index]]]
  ancestry_rectangles[[index]] <- data.frame(
    xmin = data$facet_position - 0.5,
    xmax = data$facet_position + 0.5,
    ymin = bottom,
    ymax = bottom + values,
    component = paste0("S", index),
    subspecies_label = data$subspecies_label
  )
  bottom <- bottom + values
}
ancestry_rectangles <- do.call(rbind, ancestry_rectangles)
ancestry_rectangles$component <- factor(
  ancestry_rectangles$component, levels = names(ancestry_colors)
)

region_annotations <- do.call(
  rbind,
  lapply(subspecies_levels, function(subspecies_name) {
    do.call(
      rbind,
      lapply(region_levels, function(region_name) {
        subset <- data[
          data$subspecies_label == subspecies_name &
            data$geographic_region == region_name,
        ]
        if (nrow(subset) == 0L) {
          return(NULL)
        }
        data.frame(
          xmin = min(subset$facet_position) - 0.5,
          xmax = max(subset$facet_position) + 0.5,
          x = mean(range(subset$facet_position)),
          label = unname(region_labels[region_name]),
          label_y = ifelse(
            match(region_name, region_levels) %% 2 == 0, 1.120, 1.078
          ),
          color = unname(region_colors[region_name]),
          subspecies_label = factor(subspecies_name, levels = subspecies_levels)
        )
      })
    )
  })
)
# Geographic-region annotations are shown only for the cultivated Europaea
# facet; the small Sylvestris facet is intentionally left unannotated.
region_annotations <- region_annotations[
  region_annotations$subspecies_label == "Europaea",
]

base_size <- 9.8
plot <- ggplot() +
  geom_rect(
    data = ancestry_rectangles,
    aes(
      xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax,
      fill = component
    ),
    linewidth = 0
  ) +
  geom_rect(
    data = region_annotations,
    aes(
      xmin = xmin, xmax = xmax, ymin = 1.025, ymax = 1.050,
      fill = I(color)
    ),
    linewidth = 0,
    show.legend = FALSE
  ) +
  geom_text(
    data = region_annotations,
    aes(x = x, y = label_y, label = label),
    size = 2.6,
    vjust = 0.5
  ) +
  facet_grid(
    cols = vars(subspecies_label),
    scales = "free_x",
    switch = "x"
  ) +
  scale_fill_manual(values = ancestry_colors, drop = FALSE) +
  scale_x_continuous(expand = expansion(mult = 0.005)) +
  scale_y_continuous(
    breaks = c(0, 0.25, 0.50, 0.75, 1.00),
    labels = c("0.00", "0.25", "0.50", "0.75", "1.00"),
    expand = expansion(mult = 0)
  ) +
  coord_cartesian(ylim = c(0, 1.15), clip = "off") +
  labs(x = NULL, y = "Ancestry proportion", fill = NULL) +
  guides(fill = guide_legend(ncol = 1, byrow = TRUE)) +
  theme_classic(base_size = base_size, base_family = "sans") +
  theme(
    axis.line.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.text.x = element_blank(),
    axis.title.y = element_text(margin = margin(r = 5)),
    axis.line.y = element_line(linewidth = 0.55),
    axis.ticks.y = element_line(linewidth = 0.45),
    panel.spacing.x = grid::unit(3.5, "mm"),
    strip.placement = "outside",
    strip.background = element_blank(),
    strip.text.x.bottom = element_text(
      face = "bold", size = base_size, margin = margin(t = 4)
    ),
    legend.position = "right",
    legend.direction = "vertical",
    legend.key.size = grid::unit(3.6, "mm"),
    legend.spacing.y = grid::unit(1.0, "mm"),
    legend.text = element_text(size = base_size * 0.9),
    plot.margin = margin(t = 14, r = 4, b = 5, l = 6)
  )

dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
ggsave(
  paste0(output_stem, ".pdf"),
  plot = plot,
  device = grDevices::cairo_pdf,
  width = 183 / 25.4,
  height = 3.35,
  units = "in",
  bg = "white"
)
ggsave(
  paste0(output_stem, ".png"),
  plot = plot,
  width = 183 / 25.4,
  height = 3.35,
  units = "in",
  dpi = 600,
  bg = "white"
)

message("Wrote ", output_stem, ".pdf")
message("Wrote ", output_stem, ".png")

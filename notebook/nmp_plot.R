library(tidyverse)
library(ggthemes)
library(ggplot2)

setwd("D:\\PhD_studies\\Olive_STR")

# Read ancestry proportions
prop <- read.table(".\\output_file\\dnmf_pos13.csv", sep = ",", header = TRUE)
d <- dist(prop[,-ncol(prop)])
hc <- hclust(d)
prop$sample <- factor(prop$sample, levels = prop$sample[hc$order])

# Reshape to long format for ggplot2
prop_long <- prop %>%
  pivot_longer(
    cols = starts_with("X"),
    names_to   = "Component",
    values_to  = "Proportion",
    names_transform = list(
      Component = \(x) paste0("S", as.integer(sub("^X", "", x)) + 1)
    )
  )

# Geographical_Locations
geo <- read.csv("supp1.csv", sep = ",", header = TRUE)
geo <- geo %>% filter(ID %in% prop$sample) %>% distinct(ID, .keep_all = TRUE)
geo$Subspecies <- sub(".*\\.", "", geo$Subspecies)

prop_long <- merge(prop_long, geo, by.x = "sample", by.y = "ID")

prop_long$Component <- factor(prop_long$Component, 
            levels = paste("S", seq(1,13), sep = ""))
prop_long <- prop_long %>%
  mutate(Province = ifelse(Country == "Iran", Provience, Country)) 
prop_long$Subspecies <- factor(prop_long$Subspecies,
                   levels = c(" europaea", " Sylvestris", " Cuspidata"))

write.csv(prop_long, "prop_long.csv")
prop_long

prop_subsp <- prop_long %>% filter(Subspecies == " europaea")
prop_sub <- prop_subsp %>% filter(Country != "Iran")
prop_sub$Province <- factor(prop_sub$Province,
                levels = c("Spain", "Italy",
              "Greece", "Syria", "Turkey", "Israel", "Lebanon", 
                   "Algeria", "Albania", "Tunisia"))

prop_ir <- prop_subsp %>% filter(Country == "Iran")
prop_ir$Province <- factor(prop_ir$Province,
               levels = c("Golestan","Tarom",
                "Khorasan-e-jonoubi","Kermanshah", "Lorestan",
                "Ilam", "Yazd", "Khoor va Biabanak", "Khuzestan",
                "Kohgiluyeh", "Fars", "Boushehr"
                ))

color_pale1 <- tableau_color_pal("Miller Stone")(11) 
color_pale2 <- tableau_color_pal("Superfishel Stone")(10)[7]
color_pale <- c(color_pale1, color_pale2, tableau_color_pal("Superfishel Stone")(10)[4])

ggplot(prop_subsp, aes(x = sample, y = Proportion, fill = Component)) +
  geom_bar(stat = "identity") +
  facet_grid(. ~ Province,
             scales = "free_x",
             space = "free_x") +
  scale_fill_manual(values = color_pale) +
  theme(
    text=element_text(size=20),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    panel.grid = element_blank(),
    panel.spacing = unit(0.1, "lines"),
    panel.background = element_blank(),
    strip.background.x = element_blank(),
    strip.text.x.top = element_text(
      size = 12, #15
      angle = 45,
      hjust = 0,
      margin = margin(b = -35, unit = "pt")), #20
    strip.clip = "off",
    plot.margin = margin(t = 50, unit = "pt"),
    legend.title=element_blank()
  ) +
  labs(#title = "NMF Admixture Plot", 
    x = "", 
    y = "")


 
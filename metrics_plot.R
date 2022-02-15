require('tidyverse')
require('dplyr')

scores <- read.csv('scores.csv')

latent_dim <- scores[1,2]

scores <- subset(scores, select = -c(latent_dim) ) %>% 
  gather(metric, value, -classifier, -dataset)
  
plot <- ggplot(scores, mapping = aes(x = classifier)) + 
  geom_bar(mapping = aes(y = value, fill = metric), color='black', stat ='identity', position="dodge")  + 
  facet_wrap(facets = vars(dataset), ncol = 1) + 
  ggtitle("latent_dim=", toString(latent_dim))

ggsave("scores.png")
library(dplyr)  
library(stringr)
library(readr)
library(quanteda) 
library(ggplot2)  
library(caret)
library(data.table)
library(randomForest)
library(glmnet)
library(Matrix)
library(quanteda.textstats)  

#Data Preprocessing
#Load data
df <- read_csv("twitter_data/ExtractedTweets.csv")

#Ensure document names are unique
df$doc_id <- seq_len(nrow(df))

#Check for missing data
print("Missing data situation:")
print(colSums(is.na(df)))

#Remove rows with missing values
df <- na.omit(df)

#Remove 'Handle' column and 'doc_id' column
df <- df %>% select(-Handle, -doc_id)

#Text cleaning 
clean_text <- function(text) {
  text <- as.character(text)  #Ensure input is character type
  text <- str_remove_all(text, "^RT[\\s]+")  #Remove retweets
  text <- str_remove_all(text, "http\\S+")  #Remove URLs
  text <- str_remove_all(text, "@[^\\s]+")  #Remove mentions
  text <- str_remove_all(text, "\\w+…")  # Remove truncated words like "word..."
  text <- str_remove_all(text, "[^\\w\\s]")  #Remove special characters and punctuation
  text <- str_remove_all(text, "\\d+")  #Remove all numbers
  text <- tolower(text)  # Convert to lowercase
  text <- str_trim(str_squish(text))  #Remove excess spaces
  
#Obtain a list of stopwords and remove them
  stop_words <- stopwords("en")
  text <- str_replace_all(text, paste0("\\b(", paste(stop_words, collapse = "|"), ")\\b"), "")
  
  text <- str_squish(text)  #Clean excess spaces again
  
  return(text)
}

#Ensure the Tweet column is a character vector
df$Tweet <- as.character(df$Tweet)

#Apply preprocessing function to all tweets
df$cleaned_string <- sapply(df$Tweet, clean_text)

#Check for missing data after preprocessing
print("Missing data situation after preprocessing:")
print(colSums(is.na(df)))

#Remove rows with missing values
df <- na.omit(df)

#Save preprocessed data to a new CSV file
write_csv(df, "twitter_data/twitter_data_after_preprocess.csv")
print("Preprocessing completed")

#Moral Foundation Frequency Analysis
#Load preprocessed data
df <- read_csv("twitter_data/twitter_data_after_preprocess.csv")  #Load processed tweet data
print(paste("Total number of rows:", nrow(df)))  #Print the total number of rows in the data

#Print data statistics based on political party
print(paste("Democrats:", sum(df$Party == "Democrat")))  #Print the number of rows belonging to Democrats
print(paste("Republicans:", sum(df$Party == "Republican")))  #Print the number of rows belonging to Republicans

#Load MFD dictionary file and build dictionary
mfd_file <- read_lines('mfd2.0.dic')  #Read the MFD dictionary file
mfd_dict <- list()  #Initialize an empty list to build the final dictionary
category_lines <- mfd_file[grep("^\\d+\\s", mfd_file)]  #Extract lines containing category numbers
categories <- str_split_fixed(category_lines, "\\s+", 2)  #Split category lines by space
category_names <- setNames(categories[, 2], categories[, 1])  #Create a list with numbers as names and categories as values
word_lines <- mfd_file[grep("^[a-zA-Z]", mfd_file)]  #Extract lines containing words
words <- str_split(word_lines, "\\s+(?=[^\\s]+$)", n = 2, simplify = TRUE)  #Split at the last space in each line
words <- words[nchar(words[, 2]) > 0, ]  #Ensure the second part after splitting contains characters
for (i in 1:nrow(words)) {
  word <- words[i, 1]  #Word
  category_number <- words[i, 2]  #Category number
  if (category_number %in% names(category_names)) {
    category_name <- category_names[[category_number]]  #Get the category name
    if (!category_name %in% names(mfd_dict)) {
      mfd_dict[[category_name]] <- c()  #If the category does not exist in the dictionary, create an empty vector
    }
    mfd_dict[[category_name]] <- c(mfd_dict[[category_name]], word)  #Add the word to the corresponding category
  }
}
quanteda_dict <- dictionary(mfd_dict)  #Create a quanteda dictionary object from the built dictionary

#Create a corpus
corpus <- corpus(df, text_field = "cleaned_string")  #Create a corpus using the cleaned_string field

#Process text in the corpus: create document-feature matrix
tokens <- tokens(corpus)  #Generate tokens from the corpus
dfm_tokens <- dfm(tokens)  #Create a document-feature matrix from the tokens
dfm_tweets <- dfm_lookup(dfm_tokens, dictionary = quanteda_dict)  #Filter features using the built dictionary

#Calculate the frequency of each moral foundation category
moral_freqs <- textstat_frequency(dfm_tweets)  #Calculate frequency statistics
print(moral_freqs)  #Print frequency statistics of moral foundation categories

#Split data by party
df_democrat <- df %>% filter(Party == "Democrat")  #Filter data belonging to Democrats
df_republican <- df %>% filter(Party == "Republican")  #Filter data belonging to Republicans

#Define a function to generate and save charts
plot_moral_freqs <- function(df_party, party_name, output_path) {
  corpus_party <- corpus(df_party, text_field = "cleaned_string")  #Create a corpus for each party
  tokens_party <- tokens(corpus_party)  #Generate tokens
  tokens_party <- tokens_remove(tokens_party, stopwords("en"))  #Remove stopwords
  dfm_tokens_party <- dfm(tokens_party)  #Create a document-feature matrix
  dfm_party <- dfm_lookup(dfm_tokens_party, dictionary = quanteda_dict)  #Filter features using the dictionary
  moral_freqs_party <- textstat_frequency(dfm_party)  #Calculate the moral foundation frequency for each party
  
#Print the moral foundation frequency for the current party
  print(paste("Moral Foundations Frequency for", party_name, ":"))
  print(moral_freqs_party)  #Display the frequency data
  
#Visualize the frequency of moral foundation categories
  p <- ggplot(moral_freqs_party, aes(x = reorder(feature, frequency), y = frequency, fill = feature)) +
    geom_bar(stat = "identity", color = "black", size = 0.3) +
    coord_flip() +
    labs(title = paste("Moral Foundations Frequency in", party_name, "Twitter Data"),
         x = "Moral Foundations Category",
         y = "Frequency") +
    theme_minimal(base_size = 15) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_blank(),
      plot.background = element_rect(fill = "white"),
      axis.text = element_text(size = 12),
      axis.title = element_text(size = 14)
    )
  
#Save the chart to the specified path
  ggsave(output_path, plot = p, width = 10, height = 6)
}

#Generate and save frequency charts for each party
plot_moral_freqs(df_democrat, "Democrat", "twitter_res/Democrat_Moral_Frequencies.png")  #Generate and save frequency chart for Democrats
plot_moral_freqs(df_republican, "Republican", "twitter_res/Republican_Moral_Frequencies.png")  #Generate and save frequency chart for Republicans

#Word Frequency Analysis
#Load data
data <- fread("twitter_data/twitter_data_after_preprocess.csv", select = "cleaned_string")

#Define a list of words to filter
filter_words <- c("amp", "will", "im", "now", "can", "just", "th", "dont", "one", "thats", "youre", "also", "cant", "didnt", "still", "isnt", "must", "w", "hes", "shes")

#Tokenize using stringr and filter out unwanted words
data$words <- lapply(strsplit(data$cleaned_string, "\\s+"), function(words) {
  setdiff(words, filter_words)
})

#Convert to long format
words_long <- unlist(data$words)
word_table <- table(words_long)
word_df <- as.data.frame(word_table, responseName = "frequency")
word_df$word <- as.character(row.names(word_df))
word_df <- word_df[, c("words_long", "frequency")]  # Select and arrange columns

#Sort by word frequency
word_df <- word_df[order(-word_df$frequency), ]
top_words <- head(word_df, 50)  # Extract top 50 high-frequency words

#Plotting a bar chart
p <- ggplot(data = top_words, aes(x = reorder(words_long, -frequency), y = frequency)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(x = "Word", y = "Frequency", title = "Top Word Frequencies") +
  theme_minimal(base_size = 14) +  # Use base size to enhance readability
  theme(panel.background = element_rect(fill = "white"),  # Set panel background to white
        axis.text.x = element_text(angle = 45, hjust = 1))  # Adjust X-axis text angle

#Save the data word frequency image to a PNG file
ggsave("twitter_res/twitter_data_word_frequencies.png", plot = p, width = 10, height = 8)

#Classifier Construction
# Load processed data
df <- read.csv("twitter_data/twitter_data_after_preprocess.csv")

#Ensure df is a data.frame
df <- as.data.frame(df)

#Convert the Party column to numeric (0 and 1)
df$Party <- factor(ifelse(df$Party == "Republican", 1, 0))

#Ensure document names are unique
df$doc_id <- make.names(seq_len(nrow(df)), unique = TRUE)

#Load MFD dictionary file and build dictionary
mfd_file <- readLines('mfd2.0.dic')
mfd_dict <- list()

#Extract category labels and names
category_lines <- mfd_file[grep("^\\d+\\s", mfd_file)]
categories <- str_split_fixed(category_lines, "\\s+", 2)
category_names <- setNames(categories[, 2], categories[, 1])

#Extract words and their corresponding categories
word_lines <- mfd_file[grep("^[a-zA-Z]", mfd_file)]
words <- str_split(word_lines, "\\s+(?=[^\\s]+$)", n = 2, simplify = TRUE)  #Use regex to split only at the last space

#Build dictionary
for (i in 1:nrow(words)) {
  word <- words[i, 1]
  category_number <- words[i, 2]
  if (category_number %in% names(category_names)) {
    category_name <- category_names[[category_number]]
    if (!category_name %in% names(mfd_dict)) {
      mfd_dict[[category_name]] <- c()
    }
    mfd_dict[[category_name]] <- c(mfd_dict[[category_name]], word)
  }
}

#Create a dictionary for text quantification analysis
quanteda_dict <- dictionary(mfd_dict)

#Create a corpus and set unique document names
corpus <- corpus(df, text_field = "cleaned_string", docid_field = "doc_id")
tokens <- tokens(corpus)
dfm_tokens <- dfm(tokens)

#Create Document-Feature Matrix (DFM)
dfm_tweets <- dfm_lookup(dfm_tokens, dictionary = quanteda_dict)

#Prepare training and test data
set.seed(0)
train_indices <- createDataPartition(df$Party, p = 0.8, list = FALSE)
train_data <- dfm_tweets[train_indices, ]
test_data <- dfm_tweets[-train_indices, ]

#Convert to dataframe and remove constant features
train_data_df <- as.data.frame(as.matrix(train_data))
constant_features <- sapply(train_data_df, function(x) length(unique(x)) == 1)
train_data_df <- train_data_df[, !constant_features]

#Remove rows that are all zeros
row_zeros <- apply(train_data_df, 1, function(x) all(x == 0))
train_data_df <- train_data_df[!row_zeros, ]

#Identify columns where the number of 1s is greater than 100
col_ones <- apply(train_data_df, 2, function(x) sum(x == 1) > 100)
train_data_df <- train_data_df[, col_ones]

#Apply the same column selection to test data
test_data_df <- as.data.frame(as.matrix(test_data))
test_data_df <- test_data_df[, col_ones] # Apply the same column filtering

#Logistic regression method
train_party <- df$Party[train_indices][!row_zeros]
print('Start training with logistic regression')
logistic_model <- glm(train_party ~ ., data = train_data_df, family = binomial())

#Predict on test data
test_party <- df$Party[-train_indices]
prob_predictions <- predict(logistic_model, newdata = test_data_df, type = "response")
class_predictions <- ifelse(prob_predictions > 0.5, 1, 0)
predictions <- as.factor(class_predictions)

#Output results on test data
confusion_matrix <- confusionMatrix(predictions, test_party)
print(confusion_matrix)

#Random Forest method
train_party <- df$Party[train_indices][!row_zeros]
print('Start training with Random Forest')
rf_model <- randomForest(x = train_data_df, y = train_party, ntree = 100, importance = TRUE)

#Predict on test data
test_data_df <- as.data.frame(as.matrix(test_data))
test_data_df <- test_data_df[, names(train_data_df)]
test_party <- df$Party[-train_indices]
predictions <- predict(rf_model, newdata = test_data_df)

#Output results on test data
confusion_matrix <- confusionMatrix(predictions, test_party)
print(confusion_matrix)

#Cross-validation
#Save the logistic regression model
saveRDS(logistic_model, file = "twitter_res/logistic_model.rds")

#Save the random forest model
saveRDS(rf_model, file = "twitter_res/rf_model.rds")

#Load Reddit data
reddit_df <- read.csv("reddit_data/reddit_data_after_preprocess.csv")
reddit_df <- as.data.frame(reddit_df)

#Convert the Party column to numeric
reddit_df$Party <- factor(ifelse(reddit_df$Party == "Republican", 1, 0))

#Create unique names for documents
reddit_df$doc_id <- make.names(seq_len(nrow(reddit_df)), unique = TRUE)

#Build a corpus using the same MFD dictionary
reddit_corpus <- corpus(reddit_df, text_field = "cleaned_string", docid_field = "doc_id")
reddit_tokens <- tokens(reddit_corpus)
reddit_dfm_tokens <- dfm(reddit_tokens)

#Find features in the dictionary
reddit_dfm_tweets <- dfm_lookup(reddit_dfm_tokens, dictionary = quanteda_dict)

#Convert to a dataframe and apply previous column selection
reddit_data_df <- as.data.frame(as.matrix(reddit_dfm_tweets))
reddit_data_df <- reddit_data_df[, col_ones]  #Apply previously determined valid columns

#Remove rows that are all zeros
reddit_row_zeros <- apply(reddit_data_df, 1, function(x) all(x == 0))
reddit_data_df <- reddit_data_df[!reddit_row_zeros, ]

#Load saved models
loaded_logistic_model <- readRDS("twitter_res/logistic_model.rds")
loaded_rf_model <- readRDS("twitter_res/rf_model.rds")

#Predict using the logistic regression model
reddit_prob_predictions <- predict(loaded_logistic_model, newdata = reddit_data_df, type = "response")
reddit_class_predictions <- ifelse(reddit_prob_predictions > 0.5, 1, 0)
reddit_predictions_logistic <- as.factor(reddit_class_predictions)

#Predict using the random forest model
reddit_predictions_rf <- predict(loaded_rf_model, newdata = reddit_data_df)

#Calculate accuracy of the logistic regression model
correct_logistic <- sum(reddit_predictions_logistic == reddit_df$Party[!reddit_row_zeros])
accuracy_logistic <- correct_logistic / length(reddit_predictions_logistic)
print(paste("Accuracy of Logistic Regression:", accuracy_logistic))

#Calculate accuracy of the random forest model
correct_rf <- sum(reddit_predictions_rf == reddit_df$Party[!reddit_row_zeros])
accuracy_rf <- correct_rf / length(reddit_predictions_rf)
print(paste("Accuracy of Random Forest:", accuracy_rf))



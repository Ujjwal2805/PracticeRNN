package org.example;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.List;



public class QADataset {
    private static List<List<String>> tokenizedQuestions = new ArrayList<>();
    private static List<List<String>> tokenizedAnswers = new ArrayList<>();

    public static void main(String[] args) throws Exception {
        // Create a list of question-answer pairs
        List<QAPair> qaPairs = new ArrayList<>();
        qaPairs.add(new QAPair("What is the capital of France?", "Paris"));
        qaPairs.add(new QAPair("Who wrote Romeo and Juliet?", "William Shakespeare"));
        qaPairs.add(new QAPair("What is the largest planet in our solar system?", "Jupiter"));

        // Create a tokenizer to preprocess text
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new TokenPreProcess() {
            @Override
            public String preProcess(String token) {
                return token.toLowerCase(); // Convert text to lowercase
            }
        });

        // Create a SentenceIterator to iterate through questions and answers
       // SentenceIterator iterator = new BasicLineIterator(getTextFromQA(qaPairs));
            SentenceIterator iterator=new BasicLineIterator(new File("C:\\Users\\Ujjwal\\Desktop\\PracticeRNN\\src\\main\\resources\\Data - Sheet1.csv"));
        System.out.println("The tokenized Output is : "+ tokenizerFactory);
        //iterator.setTokenizerFactory(tokenizerFactory);
        tokenizerFactory.setTokenPreProcessor((TokenPreProcess) tokenizerFactory);


        // Define the word vectors for text encoding (you'll need pre-trained word vectors)
        String wordVectorsPath = "path/to/word/vectors/model"; // Replace with your word vectors model path
        WordVectorSerializer wordVectorSerializer = (WordVectorSerializer) WordVectorSerializer.loadStaticModel(new File(wordVectorsPath));

        // Tokenize and store the data in the shared lists
        while (iterator.hasNext()) {
            String text = iterator.nextSentence();
            Tokenizer tokenizer = tokenizerFactory.create(text);
            List<String> tokens = tokenizer.getTokens();

            if (tokens.isEmpty()) {
                continue; // Skip empty sentences
            }

            // Separate questions and answers based on your dataset structure
            if (isQuestion(text)) {
                tokenizedQuestions.add(tokens);
            } else {
                tokenizedAnswers.add(tokens);
            }
        }
    }

    // Helper function to concatenate questions and answers for tokenization
    private static String getTextFromQA(List<QAPair> qaPairs) {
        StringBuilder text = new StringBuilder();
        for (QAPair qaPair : qaPairs) {
            text.append(qaPair.getQuestion()).append(" ").append(qaPair.getAnswer()).append("\n");
        }
        return text.toString();
    }

    // Helper function to determine if a text is a question (customize as needed)
    private static boolean isQuestion(String text) {

        return text.contains("?");
    }

    // Getter methods to access tokenized data
    public static List<List<String>> getTokenizedQuestions() {
        return tokenizedQuestions;
    }

    public static List<List<String>> getTokenizedAnswers() {
        return tokenizedAnswers;
    }
}

// Class to represent question-answer pairs
class QAPair {
    private final String question;
    private final String answer;

    public QAPair(String question, String answer) {
        this.question = question;
        this.answer = answer;
    }

    public String getQuestion() {
        return question;
    }

    public String getAnswer() {
        return answer;
    }
}

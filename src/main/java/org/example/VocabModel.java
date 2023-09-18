package org.example;

//import org.apache.spark.ml.feature.Word2Vec;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;

import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileNotFoundException;
import java.util.Scanner;

public class VocabModel {

    public static void TrainModel(String filePath, String modelSavePath) throws FileNotFoundException {

        SentenceIterator iter= new BasicLineIterator(filePath);
        TokenizerFactory t=new DefaultTokenizerFactory();

        t.setTokenPreProcessor(new CommonPreprocessor());

        Logger log= LoggerFactory.getLogger(VocabModel.class);

        Word2Vec vec=new Word2Vec.Builder()
                .iterations(5)
                .layerSize(10)
                .learningRate(0.01)
                .minWordFrequency(1)
                .seed(123)
                .epochs(10)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();

        vec.fit();

        WordVectorSerializer.writeWord2VecModel(vec,modelSavePath);


    }


    public static Word2Vec loadModel(String modelPath){
        org.deeplearning4j.models.word2vec.Word2Vec
                word2Vec=WordVectorSerializer.readWord2VecModel(modelPath);

        return word2Vec;
    }

    public static void main(String Args[]) throws FileNotFoundException {

        String filepath="C:\\Users\\Ujjwal\\Desktop\\PracticeRNN\\src\\main\\resources\\word2VecTrainer.txt";
        String modelPath="C:\\Users\\Ujjwal\\Desktop\\PracticeRNN\\src\\main\\resources\\word2VecModel.zip";

        TrainModel(filepath,modelPath);
        Word2Vec vec =loadModel(modelPath);

        Scanner sc=new Scanner(System.in);


        while(true){
            String text=sc.nextLine();
            System.out.println("Word Vector is: "+ vec.getWordVector(text));

        }


    }






}

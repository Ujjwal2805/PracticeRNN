package org.example;

import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.List;

public class QAModel {
    public static void main(String[] args) {
        int batchSize = 1; // Adjust batch size as needed

        // Access tokenized questions and answers from QADataset
        List<List<String>> tokenizedQuestions = QADataset.getTokenizedQuestions();
        List<List<String>> tokenizedAnswers = QADataset.getTokenizedAnswers();

        // Convert tokenized questions and answers to INDArrays (similar to previous code)
        INDArray questionsArray = convertTokenListToOneHot(tokenizedQuestions,5);
        INDArray answersArray = convertTokenListToOneHot(tokenizedAnswers,5);



        // Define your input and output datasets (e.g., using DataSet or MultiDataSet)
        DataSet dataSet = new org.nd4j.linalg.dataset.DataSet(questionsArray, answersArray);

        // Define your training data iterator using dataSet
        DataSetIterator dataSetIterator = new ListDataSetIterator<>(dataSet.asList(), batchSize);




        int inputSize = 100; // Replace with the size of your word vectors
        int outputSize = 1; // Number of output classes (1 for binary classification)
        int numEpochs = 10;
       // int batchSize = 1;
        int numHiddenNodes = 20;

        // Define the neural network configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                //.iterations(1)
                //.learningRate(0.01)
                .updater(Updater.NESTEROVS)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(inputSize)
                        .nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .nIn(numHiddenNodes)
                        .nOut(outputSize)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        // Load your dataset here using a DataSetIterator
        // DataSetIterator dataSetIterator = ...

        // Train the model
        for (int i = 0; i < numEpochs; i++) {
            model.fit(dataSetIterator);
        }

        // Evaluate the model
        Evaluation evaluation = new Evaluation(outputSize);
        while (dataSetIterator.hasNext()) {
            org.nd4j.linalg.dataset.DataSet dataSets = dataSetIterator.next();
            INDArray output = model.output(dataSets.getFeatures());
            evaluation.eval(dataSets.getLabels(), output);
        }
        System.out.println(evaluation.stats());


    }


    public static INDArray convertTokenListToOneHot (List<List<String>> tokenizedText, int vocabSize) {
        // Initialize an INDArray to store one-hot encoded vectors
        int numExamples = tokenizedText.size();
        int maxTokens = tokenizedText.stream().mapToInt(List::size).max().orElse(0);
        INDArray oneHotArray = Nd4j.zeros(numExamples, maxTokens, vocabSize);

        // Convert each token to a one-hot encoded vector
        for (int i = 0; i < numExamples; i++) {
            List<String> tokens = tokenizedText.get(i);
            for (int j = 0; j < tokens.size(); j++) {
                String token = tokens.get(j);
                int tokenIndex =0; /* Map token to an index in your vocabulary */
                oneHotArray.putScalar(new int[]{i, j, tokenIndex}, 1.0);
            }
        }

        return oneHotArray;
    }

}


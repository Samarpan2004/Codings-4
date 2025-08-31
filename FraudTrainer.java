package com.sam.fraud;


import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.SerializationHelper;


import java.io.File;


public class FraudTrainer {
public static void main(String[] args) throws Exception {
CSVLoader loader = new CSVLoader();
loader.setSource(new File("data/transactions.csv"));
Instances data = loader.getDataSet();
// set class index to last column
data.setClassIndex(data.numAttributes() - 1);


RandomForest rf = new RandomForest();
rf.setNumIterations(100);
rf.buildClassifier(data);


SerializationHelper.write("models/randomforest-fraud.model", rf);
System.out.println("Model trained and saved.");
}
}
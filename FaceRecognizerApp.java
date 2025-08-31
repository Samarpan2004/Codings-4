package com.sam.face;


import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.face.LBPHFaceRecognizer;


public class FaceRecognizerApp {
static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }
public static void main(String[] args) {
String model = "model/lbph-model.yml";
LBPHFaceRecognizer recognizer = LBPHFaceRecognizer.create();
recognizer.read(model);
String testImage = "data/test/person1.jpg";
Mat gray = Imgcodecs.imread(testImage, Imgcodecs.IMREAD_GRAYSCALE);
int[] label = new int[1];
double[] confidence = new double[1];
recognizer.predict(gray, label, confidence);
System.out.println("Predicted label: " + label[0] + " confidence=" + confidence[0]);
}
}
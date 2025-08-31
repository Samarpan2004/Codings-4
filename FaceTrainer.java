package com.sam.face;


import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.face.LBPHFaceRecognizer;
import org.opencv.face.Face;


import java.io.File;
import java.util.ArrayList;


public class FaceTrainer {
static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }


public static void train(String datasetPath, String outputModelPath) {
File ds = new File(datasetPath);
File[] persons = ds.listFiles(File::isDirectory);
ArrayList<Mat> images = new ArrayList<>();
ArrayList<Integer> labels = new ArrayList<>();
int label = 0;
for (File person : persons) {
for (File imgFile : person.listFiles()) {
Mat img = Imgcodecs.imread(imgFile.getAbsolutePath(), Imgcodecs.IMREAD_GRAYSCALE);
images.add(img);
labels.add(label);
}
label++;
}
LBPHFaceRecognizer recognizer = LBPHFaceRecognizer.create();
recognizer.train(images, org.opencv.core.MatOfInt.fromArray(labels.stream().mapToInt(i->i).toArray()));
recognizer.save(outputModelPath);
}
}
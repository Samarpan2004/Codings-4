package com.sam.face;


import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;


public class FaceDetector {
private CascadeClassifier cascade;
public FaceDetector(String cascadePath){
System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
cascade = new CascadeClassifier(cascadePath);
}
public Rect[] detectFaces(String imagePath){
Mat img = Imgcodecs.imread(imagePath);
Mat gray = new Mat();
Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY);
MatOfRect faces = new MatOfRect();
cascade.detectMultiScale(gray, faces);
return faces.toArray();
}
}
/*
 * dnn.cpp
 *
 *  Created on: 7 juin 2017
 *      Author: tux
 */

#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::dnn;

void getMaxClass(Blob &probBlob, int *classId, double *classProb);

vector<String> readClassNames(const char *filename = "synset_words.txt")
{
    vector<String> classNames;
    ifstream fp(filename);

    string name;

    while (!fp.eof())
    {
        getline(fp, name);

        if (name.length())
        {
            classNames.push_back(name.substr(name.find(' ')+1));
        }
    }

    fp.close();
    return classNames;
}

int main()
{
    String modelTxt = "bvlc_googlenet.prototxt";
    String modelBin = "bvlc_googlenet.caffemodel";
    String imageFile = "Img/monkey.jpg";

    stringstream a;

    Net net;
    Mat img;

    Blob prob;
    Blob inputBlob;

    vector<String> classNames;

    Ptr<Importer> importer;

    float proba;
    int classId;
    double classProb;

    try
    {
        importer = createCaffeImporter(modelTxt, modelBin);
    }
    catch (const cv::Exception &err)
    {
        cerr << err.msg << endl;
    }

    importer->populateNet(net);
    importer.release();

    img = imread(imageFile);
    resize(img, img, Size(224, 224));
    inputBlob = Blob::fromImages(img);

    net.setBlob(".data", inputBlob);
    net.forward();

    prob = net.getBlob("prob");
    getMaxClass(prob, &classId, &classProb);

    classNames = readClassNames();

    proba = classProb*100;
    a << proba;

    rectangle(img, Point(0,0), Point(img.cols, img.rows-180), Scalar(0,0,0),-1);
    putText(img, "Type: ", Point(15,15),1, 1, Scalar(255,255,255),1,8);
    putText(img, classNames.at(classId), Point(65,15),1, 1, Scalar(255,255,255),1,8);
    putText(img, "Proba: ", Point(15,35),1, 1, Scalar(255,255,255),1,8);
    putText(img, a.str(), Point(70,35),1, 1, Scalar(255,255,255),1,8);
    putText(img, " %", Point(130,35),1, 1, Scalar(255,255,255),1,8);

    imshow("Image", img);
    waitKey(0);

    return 0;
}

void getMaxClass(Blob &probBlob, int *classId, double *classProb)
{
    Mat probMat = probBlob.matRefConst().reshape(1, 1);

    Point classNumber;

    minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);

    *classId = classNumber.x;
}


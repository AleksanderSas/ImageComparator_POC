// See https://aka.ms/new-console-template for more information
using Emgu.CV.Util;
using Emgu.CV;
using Emgu.CV.ML;

Console.WriteLine("Hello, World!");

string directory = "C:\\Projects\\watches\\";
Mat img1 = CvInvoke.Imread(directory + "Alt 1.jpg");
Mat img2 = CvInvoke.Imread(directory + "Alt 2.jpg");
Mat img3 = CvInvoke.Imread(directory + "Alt2 1.jpg");
Mat img4 = CvInvoke.Imread(directory + "Alt2 2.jpg");
Mat img5 = CvInvoke.Imread(directory + "SWE 1.jpeg");
Mat img6 = CvInvoke.Imread(directory + "SWE 2.jpeg");
Mat stolen = CvInvoke.Imread(directory + "Stolen 2.jpeg");


var desc1 = GetFature(img1);
var desc2 = GetFature(img2);
var desc3 = GetFature(img3);
var desc4 = GetFature(img4);
var desc5 = GetFature(img5);
var desc6 = GetFature(img6);
var stolenFeatures = GetFature(stolen);

Console.WriteLine("Alt 1.jpg:   " + stolenFeatures.Similarity(desc1));
Console.WriteLine("Alt 2.jpg:   " + stolenFeatures.Similarity(desc2));
Console.WriteLine("Alt2 1.jpg:  " + stolenFeatures.Similarity(desc3));
Console.WriteLine("Alt2 2.jpg:  " + stolenFeatures.Similarity(desc4));
Console.WriteLine("SWE 1.jpg:   " + stolenFeatures.Similarity(desc5));
Console.WriteLine("SWE 2.jpg:   " + stolenFeatures.Similarity(desc6));

desc1.Similarity(desc2);

static Desc GetFature(Mat img)
{
    var vwc = new VectorOfKeyPoint();
    var descriptor = new Mat();

    var algorithm = new Emgu.CV.Features2D.KAZE();// Try other algorithms
    algorithm.DetectAndCompute(img, null, vwc, descriptor, false);

    List<(int idx, float response)> points = new List<(int idx, float response)> ();
    for (int i = 0; i < vwc.Size; i++)
    {
        var resposne = vwc[i].Response;
        points.Add((i, resposne));
        points.Sort((x, y) => Math.Sign(y.response - x.response));
    }
    
    return new Desc 
    { 
        Point = points, 
        Descriptor = descriptor 
    };
}

struct Desc
{
    public List<(int idx, float response)> Point;
    public Mat Descriptor;

    public double Similarity(Desc x)
    {
        double finalScore = 0.0;
        List<double> scores = new List<double> ();

        //try 300 most responsive points
        for(int i = 0 ; i < 300; i++) 
        {
            double sumMin = 100000;
            var row1 = Descriptor.Row(Point[i].idx);

            //find most similar point
            for (int k = 0; k < 300; k++)
            {
                var row2 = x.Descriptor.Row(x.Point[k].idx);
                double sum = VecLen(row1 - row2) / (VecLen(row1) + VecLen(row2));
                if(sumMin > sum)
                {
                    sumMin = sum;
                }
            }
            scores.Add(sumMin);
        }
        //take into account only well matching scores, skip 100 worst matches
        scores.Sort();
        for(int i = 0; i < 200; i++)
        {
            finalScore += Math.Log(1-scores[i]);
        }
        return finalScore;
    }

    static double VecLen(Mat vec)
    {
        double sum = 0;
        var tmp = vec.GetData();
        for (int n = 0; n < tmp.Length; n++)
        {
            float a = (float)tmp.GetValue(0, n);
            sum += a * a;
        }
        return Math.Sqrt(sum);
    }
}
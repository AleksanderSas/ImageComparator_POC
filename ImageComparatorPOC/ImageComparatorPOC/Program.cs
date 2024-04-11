// See https://aka.ms/new-console-template for more information
using Emgu.CV.Util;
using Emgu.CV;

List<string> files = new List<string> { "Alt 1.jpg", "Alt 2.jpg", "Alt2 1.jpg", "Alt2 2.jpg", "SWE 1.jpeg", "SWE 2.jpeg", "Stolen 2.jpeg" };
string directory = "C:\\Projects\\watches\\";

List<Desc> descriptors = files
    .Select(x => GetFature(CvInvoke.Imread(directory + x), x))
    .ToList();

//Run for all images or just for one
#if true
foreach(var d1 in descriptors)
{
    Test(descriptors, d1);
}
#else
Test(descriptors, descriptors[6]);
#endif

static void Test(List<Desc> descriptors, Desc testedImage)
{
    Console.WriteLine(testedImage.Name);
    foreach (var d2 in descriptors)
    {
        if (testedImage != d2)
        {
            Console.WriteLine($"{d2.Name}:   {testedImage.Similarity(d2)}");
        }
    }

    Console.WriteLine();
}

static Desc GetFature(Mat img, string name)
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
        Descriptor = descriptor,
        Name = name
    };
}

class Desc
{
    public List<(int idx, float response)> Point;
    public Mat Descriptor;
    public string Name;

    public double Similarity(Desc x)
    {
        double finalScore = 0.0;
        List<(double score, int matchIdx)> scores = new List<(double, int)> ();

        //try 300 most responsive points
        for (int i = 0 ; i < 300; i++) 
        {
            int bestIdx = 0;
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
                    bestIdx = k;
                }
            }
            scores.Add((sumMin, bestIdx));
        }

        int[] hits = new int[300];
        //take into account only well matching scores, skip 100 worst matches
        scores.Sort( (x, y) => x.score.CompareTo(y.score));
        for (int i = 0; i < 200; i++)
        {
            hits[scores[i].matchIdx]++;
        }
        for (int i = 0; i < 200; i++)
        {
            // ln(x*y*z) = ln(x) + ln(y) + ln(z)
            finalScore += Math.Log(1 - scores[i].score);
            // ln(1/x) = -ln(x)
            finalScore -= Math.Log(hits[scores[i].matchIdx]);
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
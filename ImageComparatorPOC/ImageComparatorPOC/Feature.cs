using Emgu.CV;
using Emgu.CV.Util;
using System.Drawing;

namespace ImageComparatorPOC;

class Feature
{
    public required List<(int idx, float response)> Point;
    public required Mat Descriptor;
    public required string Name;

    public static Feature GetFature(Mat imgIn, string name, float threshold = 0.001f)
    {
        try
        {
            Mat img = new Mat();
            if (imgIn.Width > 1000)
            {
                CvInvoke.Resize(imgIn, img, new Size(740, (int)(740.0 / imgIn.Width * imgIn.Height)));
            }
            else
            {
                img = imgIn;
            }

            var vwc = new VectorOfKeyPoint();
            var descriptor = new Mat();

            var algorithm = new Emgu.CV.Features2D.KAZE(threshold: threshold);// Try other algorithms
            algorithm.DetectAndCompute(img, null, vwc, descriptor, false);

            List<(int idx, float response)> points = new List<(int idx, float response)>();
            for (int i = 0; i < vwc.Size; i++)
            {
                var resposne = vwc[i].Response;
                points.Add((i, resposne));
            }
            points.Sort((x, y) => Math.Sign(y.response - x.response));

            //Console.WriteLine($"Read [points {points.Count}]: {name}");
            return new Feature
            {
                Point = points,
                Descriptor = descriptor,
                Name = name
            };
        }
        catch (Exception ex)
        {
            Console.WriteLine($"cannot read {name} due to : {ex}");
            return null;
        }
    }

    public double Similarity(Feature x, int comparePoints = 300, int bestPoints = 200)
    {
        try
        {
            double finalScore = 0.0;
            List<(double score, int matchIdx)> scores = new List<(double, int)>();

            //try N most responsive points
            for (int i = 0; i < comparePoints; i++)
            {
                int bestIdx = 0;
                double sumMin = 100000;
                var row1 = Descriptor.Row(Point[i].idx);
                var row1Len = VecLen(row1);

                //find most similar point
                for (int k = 0; k < comparePoints; k++)
                {
                    var row2 = x.Descriptor.Row(x.Point[k].idx);
                    double sum = VecLen(row1 - row2) / (row1Len + VecLen(row2));
                    if (sumMin > sum)
                    {
                        sumMin = sum;
                        bestIdx = k;
                    }
                }
                scores.Add((sumMin, bestIdx));
            }

            int[] hits = new int[comparePoints];
            //take into account only well matching scores, skip K worst matches
            scores.Sort((x, y) => x.score.CompareTo(y.score));
            for (int i = 0; i < bestPoints; i++)
            {
                hits[scores[i].matchIdx]++;
            }
            for (int i = 0; i < bestPoints; i++)
            {
                // ln(x*y*z) = ln(x) + ln(y) + ln(z)
                finalScore += Math.Log(1 - scores[i].score);
                // ln(1/x) = -ln(x)
                finalScore -= Math.Log(hits[scores[i].matchIdx]);
            }

            return finalScore;
        }
        catch (Exception e)
        {
            //Console.Error.WriteLine($"Error for {Name}:  {e}");
            return 0.0;
        }
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

using Emgu.CV;
using Emgu.CV.Util;
using System.Drawing;

namespace ImageComparatorPOC;

class Feature
{
    public required Mat Descriptor;
    public required string Name;
    public Mat ResisedImage;

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

            List<(int idx, float response, PointF point)> points = new List<(int idx, float response, PointF point)>();
            for (int i = 0; i < vwc.Size; i++)
            {
                var resposne = vwc[i].Response;
                var point = vwc[i].Point;
                points.Add((i, resposne, point));
            }
            points.Sort((x, y) => Math.Sign(y.response - x.response));
            if(descriptor.Rows < 300)
            {
                Console.WriteLine($"Too low interest points: {descriptor.Rows}");
                return null;
            }
            
            //((float)descriptor.Row(points[166].idx).GetData().GetValue(0, 44)) == ((float)trimmedMat.Row(166).GetData().GetValue(0, 44))
            var trimmedMat = new Mat(300, 66, Emgu.CV.CvEnum.DepthType.Cv32F, descriptor.NumberOfChannels);
            LoadDesriptor(descriptor, points, trimmedMat);

            return new Feature
            {
                Descriptor = trimmedMat,
                Name = name,
                ResisedImage = img
            };
        }
        catch (Exception ex)
        {
            Console.WriteLine($"cannot read {name} due to : {ex}");
            return null;
        }
    }

    private unsafe static void LoadDesriptor(Mat descriptor, List<(int idx, float response, PointF point)> points, Mat trimmedMat)
    {
        for (int i = 0; i < 300; i++)
        {
            var trimmedRow = trimmedMat.Row(i);
            var sourceRow = descriptor.Row(points[i].idx);

            var dp = (float*)trimmedRow.DataPointer.ToPointer();
            var sp = (float*)sourceRow.DataPointer.ToPointer();
            for (int k = 0; k < 64; k++)
            {
                *dp = *sp;
                dp++;
                sp++;
            }
            *dp = points[i].point.X;
            dp++;
            *dp = points[i].point.Y;
        }
    }

    private Geometry GetCentroid(IEnumerable<int> idxs, int len)
    {
        var x = 0.0;
        var y = 0.0;
        var data = Descriptor.GetData();
        List<double> angles = new List<double>(len);
        foreach(var idx in idxs)
        {
            x += (float)data.GetValue(idx, 64);
            y += (float)data.GetValue(idx, 65);
        }

        x /= len;
        y /= len;

        var diffSum = 0.0;
        foreach (var idx in idxs)
        {
            var (d, a) = distance(x, y, idx);
            diffSum += d;
            angles.Add(a);
        }

        return new Geometry(x, y, diffSum / len, angles);
    }

    class Geometry
    {
        public double x;
        public double y;
        public double factor;
        public List<double> angles;

        public Geometry(double x, double y, double factor, List<double> angles)
        {
            this.x = x;
            this.y = y;
            this.factor = factor;
            this.angles = angles;
        }
    }

    private (double dist, double angle) distance(double centroidX, double centroidY, int idx)
    {
        var data = Descriptor.GetData();
        var diffX = (float)data.GetValue(idx, 64) - centroidX;
        var diffY = (float)data.GetValue(idx, 65) - centroidY;
        return (Math.Sqrt(diffX * diffX + diffY * diffY), Math.Atan2(diffY, diffX));
    }

    public (double basicScore, double angleScore, double distScore) Similarity(Feature x, bool useGeometryFeatures, int comparePoints = 300, int bestPoints = 200)
    {
        try
        {
            double finalScore = 0.0;
            double angleScore = 0.0;
            double distScore = 0.0;
            List<(double score, int matchIdx, int originIdx)> scores = GetPointMapping(x, comparePoints);

            int[] hits = new int[comparePoints];
            //take into account only well matching scores, skip K worst matches

            Geometry originCentroid = null;
            Geometry MatchedCentroid = null;
            var avgAngle = 0.0;
            if (useGeometryFeatures)
            {
                originCentroid = GetCentroid(scores.Select(x => x.originIdx).Take(bestPoints), bestPoints);
                MatchedCentroid = x.GetCentroid(scores.Select(x => x.matchIdx).Take(bestPoints), bestPoints);
                for (int i = 0; i < bestPoints; i++)
                {
                    avgAngle += originCentroid.angles[i] - MatchedCentroid.angles[i];
                }
                avgAngle /= bestPoints;
            }

            for (int i = 0; i < bestPoints; i++)
            {
                hits[scores[i].matchIdx]++;
            }

            for (int i = 0; i < bestPoints; i++)
            {
                if (useGeometryFeatures)
                {
                    var (d1, a1) = distance(originCentroid!.x, originCentroid.y, scores[i].originIdx);
                    d1 /= originCentroid.factor;
                    var (d2, a2) = x.distance(MatchedCentroid!.x, MatchedCentroid.y, scores[i].matchIdx);
                    d2 /= MatchedCentroid.factor;
                    distScore += Math.Log(1 - Math.Abs(d1 - d2) / (d1 + d2));
                    angleScore += GetAngleScore(avgAngle, a1, a2);
                }

                // ln(x*y*z) = ln(x) + ln(y) + ln(z)
                finalScore += Math.Log(1 - scores[i].score);
                // ln(1/x) = -ln(x)
                finalScore -= Math.Log(hits[scores[i].matchIdx]);
            }

            return (finalScore + angleScore + distScore, angleScore, distScore);
        }
        catch (Exception e)
        {
            //Console.Error.WriteLine($"Error for {Name}:  {e}");
            return (0.0, 0.0, 0.0);
        }
    }

    public List<(double score, int matchIdx, int originIdx)> GetPointMapping(Feature x, int comparePoints)
    {
        List<(double score, int matchIdx, int originIdx)> scores = new List<(double, int, int)>();

        //try N most responsive points
        for (int i = 0; i < comparePoints; i++)
        {
            int bestIdx = 0;
            double sumMin = 100000;
            var row1 = Descriptor.Row(i);
            var row1Len = VecLen(row1, 64);

            //find most similar point
            for (int k = 0; k < comparePoints; k++)
            {
                var row2 = x.Descriptor.Row(k);
                double sum = VecLen(row1 - row2, 64) / (row1Len + VecLen(row2, 64));
                if (sumMin > sum)
                {
                    sumMin = sum;
                    bestIdx = k;
                }
            }
            scores.Add((sumMin, bestIdx, i));
        }
        scores.Sort((x, y) => x.score.CompareTo(y.score));
        return scores;
    }

    private static double GetAngleScore(double avgAngle, double a1, double a2)
    {
        a2 += avgAngle;

        if (a2 > Math.PI)
        {
            a2 -= Math.PI;
        }
        if (a2 < -Math.PI)
        {
            a2 += Math.PI;
        }

        var angD = Math.Abs(a2 - a1);
        if (angD > Math.PI)
        {
            angD = 2 * Math.PI - angD;
        }
        return Math.Log(1 - angD / System.Math.PI);
    }

    static double VecLen(Mat vec, int pointsLen)
    {
        double sum = 0;
        var tmp = vec.GetData();
        for (int n = 0; n < pointsLen; n++)
        {
            float a = (float)tmp.GetValue(0, n);
            sum += a * a;
        }
        return Math.Sqrt(sum);
    }

    public void Save(string filename)
    {
        try
        {
            var cc = Math.Max(filename.LastIndexOf('/'), filename.LastIndexOf('\\'));
            var dir = filename.Substring(0, cc);
            if(!Directory.Exists(dir))
            {
                Directory.CreateDirectory(dir);
            }
            Descriptor.Save(filename);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Cannot save file {filename} due to exception {ex}");
        }
    }

    public static Feature Load(string filename, string originFile, bool useGeometryFeature)
    {
        if(!File.Exists(filename))
        {
            return null;
        }
        try 
        {
            var descriptor = new Mat(filename, Emgu.CV.CvEnum.ImreadModes.Unchanged);
            if(!useGeometryFeature && descriptor.Cols != 66)
            {
                var zeros = Mat.Zeros(300, 66, Emgu.CV.CvEnum.DepthType.Cv32F, 1);
                Fill(zeros, descriptor);
                descriptor = zeros;
            }
            if (useGeometryFeature && descriptor.Cols != 66)
            {
                return null;
            }
            return new Feature
            { 
                Descriptor = descriptor,
                Name = originFile
            };
        }
        catch(Exception e) 
        {
            Console.WriteLine($"Cannot load from file {filename} due to exception {e}");
            return null;
        }
    }

    private unsafe static void Fill(Mat desc, Mat source)
    {
        var d = (float*)desc.DataPointer.ToPointer();
        var s = (float*)source.DataPointer.ToPointer();

        for(int i = 0; i < 300; i++)
        {
            for (int k = 0; k < 64; k++, d++, s++)
            {
                *d = *s;
            }
            d++;
            d++;
        }
    }

    public unsafe void CopyInto(Mat mat, int xShift)
    {
        var cols = ResisedImage.Cols;
        for (int y = 0; y < ResisedImage.Rows; y++)
        {
            var source = ResisedImage.Row(y);
            var destination = mat.Row(y);

            var sp = (byte*)source.DataPointer.ToPointer();
            var dp = (byte*)destination.DataPointer.ToPointer() + xShift * 3;
            for (int k = 0; k < cols * 3; k++)
            {
                *dp = *sp;
                dp++;
                sp++;
            }
        }
    }
}

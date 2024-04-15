// See https://aka.ms/new-console-template for more information
using Emgu.CV.Util;
using Emgu.CV;
using System.Drawing;
using ImageComparatorPOC;


//RunTest();
await SearchInDirectory();

static async Task SearchInDirectory()
{
    string teseed = "C:\\Projects\\watches\\SWE-production\\Stolen\\Vacheron_constatin_Overseas.jpg";
    string directory = "C:\\Projects\\watches\\SWE-production\\Vacheron Constantin";
    string[] files = Directory.GetFiles(directory);

    var testedDescriptor = GetFature(CvInvoke.Imread(teseed), "Vacheron_constatin_Overseas.jpg");

    //For quick test
    //files = files.Take(10).ToArray();

    var readImageContext = new ParallelContext { TotalCount = files.Length };
    var batches = files.ToList().Batches(4);
    var taskResults = await Task.WhenAll(batches
        .Select(x => GetFeatureAsync(x, readImageContext)));

    List<Desc> descriptors = taskResults
        .SelectMany(x => x)
        .Where(x => x != null)
        .ToList();

    Console.WriteLine();

    await TestAsync(descriptors, testedDescriptor, new ParallelContext { TotalCount = descriptors.Count });
}

static Task<List<Desc>> GetFeatureAsync(IList<string> files, ParallelContext context)
{
    return Task.Run(() => files.Select(y =>
    {
        var tmp = GetFature(CvInvoke.Imread(y), y);
        lock(context)
        {
            Console.Write($"\rRead images {++context.FinishedCount}\\{context.TotalCount}");
        }
        return tmp;
    }).ToList());
}


static void RunTest()
{
    List<string> files = new List<string> { "Alt 1.jpg", "Alt 2.jpg", "Alt2 1.jpg", "Alt2 2.jpg", "SWE 1.jpeg", "SWE 2.jpeg", "Stolen 2.jpeg" };
    string directory = "C:\\Projects\\watches\\";

    List<Desc> descriptors = files
        .Select(x => GetFature(CvInvoke.Imread(directory + x), x, 0.002f))
        .ToList();

    //Run for all images or just for one
#if true
    foreach (var d1 in descriptors)
    {
        Test(descriptors, d1);
    }
#else
    Test(descriptors, descriptors[6]);
#endif
}

static async Task TestAsync(List<Desc> descriptors, Desc testedImage, ParallelContext context)
{
    var taskResults = await Task.WhenAll(descriptors
        .Batches(15)
        .Select(x => Task.Run(() => x.Select(y => Compute(testedImage, y, context)).ToList())));

    var results = taskResults.SelectMany(x => x).ToList();
    results.Sort();

    Console.WriteLine();
    foreach (var r in results)
    {
        Console.WriteLine($"{r.Image} {r.Score}");
    }
}

static ComparisionResult Compute(Desc testedImage, Desc y, ParallelContext context)
{
    var tmp = new ComparisionResult(y.Similarity(testedImage), y.Name);

    lock(context)
    {
        Console.Write($"\rCompute {++context.FinishedCount}\\{context.TotalCount}");
    }

    return tmp;
}

static void Test(List<Desc> descriptors, Desc testedImage)
{
    Console.WriteLine(testedImage.Name);
    foreach (var d2 in descriptors)
    {
        if (testedImage != d2)
        {
            Console.WriteLine($"{d2.Name}:   {d2.Similarity(testedImage)}");
        }
    }

    Console.WriteLine();
}

static Desc GetFature(Mat imgIn, string name, float threshold = 0.001f)
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
        return new Desc
        {
            Point = points,
            Descriptor = descriptor,
            Name = name
        };
    }
    catch(Exception ex) 
    {
        Console.WriteLine($"cannot read {name} due to : {ex}");
        return null;
    }
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
            var row1Len = VecLen(row1);

            //find most similar point
            for (int k = 0; k < 300; k++)
            {
                var row2 = x.Descriptor.Row(x.Point[k].idx);
                double sum = VecLen(row1 - row2) / (row1Len + VecLen(row2));
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
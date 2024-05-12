using Emgu.CV;

namespace ImageComparatorPOC;

internal class Tester
{
    private const int ThreadNo = 6; //best results for number similar to number of physical cores
    public static void RunTest()
    {
        List<string> files = new List<string> { "Alt 1.jpg", "Alt 2.jpg", "Alt2 1.jpg", "Alt2 2.jpg", "SWE 1.jpeg", "SWE 2.jpeg", "Stolen 2.jpeg" };
        string directory = "C:\\Projects\\watches\\";

        List<Feature> descriptors = files
            .Select(x => Feature.GetFature(CvInvoke.Imread(directory + x), x, 0.002f))
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

    private static void Test(List<Feature> descriptors, Feature testedImage)
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

    public static async Task<List<ComparisionResult>> TestAsync(List<Feature> descriptors, Feature testedImage)
    {
        var results = await TestInternalAsync(descriptors, testedImage, 90, 70);
        var toTakeLimit = Math.Max(40, (results.Count / 1000) * 5);
        results = results.Where(x => x.Score < 0.0 && x.ScorePerPoint > -1.10).Take(toTakeLimit).ToList();
        //var results = await TestInternalAsync(descriptors, testedImage, 100, 80);
        //results = results.Where(x => x.Score < 0.0 && x.ScorePerPoint > -3.10).Take(60).ToList();
        return await TestInternalAsync(results.Select(x => x.OtherFeature).ToList(), testedImage, 300, 200);
    }

    private static async Task<List<ComparisionResult>> TestInternalAsync(List<Feature> descriptors, Feature testedImage, int comparePoints, int bestPoints)
    {
        if(descriptors.Count == 0)
        {
            Console.Write($"\rCompute 0\\0   Threads: 0");
            return new List<ComparisionResult>();
        }
        var context = new ParallelContext
        {
            TotalCount = descriptors.Count,
            ThreadCount = ThreadNo
        };

        var taskResultsTmp = descriptors
            .Batches(descriptors.Count / ThreadNo);

        Console.WriteLine($"Batches: {taskResultsTmp.Count}");

        var taskResults = await Task.WhenAll(
            taskResultsTmp.Select(x => Task.Run(() =>
            {
                var tmp = x.Select(y => Compute(testedImage, y, context, comparePoints, bestPoints)).ToList();
                context.Decrement();
                return tmp;
            })).ToList());

        Console.WriteLine();
        var results = taskResults.SelectMany(x => x).ToList();
        results.Sort();
        return results;
    }

    private static ComparisionResult Compute(Feature testedImage, Feature y, ParallelContext context, int comparePoints, int bestPoints)
    {
        var score = y.Similarity(testedImage, comparePoints, bestPoints);
        var tmp = new ComparisionResult(score, score / bestPoints, y.Name, y);

        var c = Interlocked.Increment(ref context.FinishedCount);
        if (c % 8 == 0)
        {
        lock (context)
        {
                Console.Write($"\rCompute {c}\\{context.TotalCount}   Threads: {context.ThreadCount}   MPI: {context.MilisPerImage()}");
            }
        }

        return tmp;
    }
}

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

    public static async Task TestAsync(List<Feature> descriptors, Feature testedImage)
    {
        var results = await TestInternalAsync(descriptors, testedImage, 100, 75);
        results = results.Where(x => x.Score < 0.0).Take(100).ToList();
        var results2 = await TestInternalAsync(results.Select(x => x.OtherFeature).ToList(), testedImage, 300, 200);

        var resultPath = $"C:\\Projects\\watches\\{testedImage.Name}.txt";
        using (StreamWriter outputFile = new StreamWriter(resultPath))
        {
            Console.WriteLine();
            foreach (var r in results2)
            {
                //Console.WriteLine($"{r.Image} {r.Score}");
                outputFile.WriteLine($"{r.Image} {r.Score}");
            }
        }
    }

    private static async Task<List<ComparisionResult>> TestInternalAsync(List<Feature> descriptors, Feature testedImage, int comparePoints, int bestPoints)
    {
        var context = new ParallelContext
        {
            TotalCount = descriptors.Count,
            ThreadCount = ThreadNo
        };

        var taskResults = await Task.WhenAll(descriptors
            .Batches(descriptors.Count / ThreadNo)
            .Select(x => Task.Run(() =>
            {
                var tmp = x.Select(y => Compute(testedImage, y, context, comparePoints, bestPoints)).ToList();
                context.ThreadCount--;
                return tmp;
            })).ToList());

        var results = taskResults.SelectMany(x => x).ToList();
        results.Sort();
        return results;
    }

    private static ComparisionResult Compute(Feature testedImage, Feature y, ParallelContext context, int comparePoints, int bestPoints)
    {
        var tmp = new ComparisionResult(y.Similarity(testedImage, comparePoints, bestPoints), y.Name, y);

        lock (context)
        {
            Console.Write($"\rCompute {++context.FinishedCount}\\{context.TotalCount}   Threads: {context.ThreadCount}");
        }

        return tmp;
    }
}

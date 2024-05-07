using Emgu.CV;

namespace ImageComparatorPOC;

internal class Tester
{
    public static void RunTest()
    {
        List<string> files = new List<string> { "Alt 1.jpg", "Alt 2.jpg", "Alt2 1.jpg", "Alt2 2.jpg", "SWE 1.jpeg", "SWE 2.jpeg", "Stolen 2.jpeg" };
        string directory = "C:\\Projects\\watches\\";

        List<Fature> descriptors = files
            .Select(x => Fature.GetFature(CvInvoke.Imread(directory + x), x, 0.002f))
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

    private static void Test(List<Fature> descriptors, Fature testedImage)
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

    public static async Task TestAsync(List<Fature> descriptors, Fature testedImage)
    {
        var threadNo = 6; //best results for number similar to number of physical cores
        var context = new ParallelContext
        {
            TotalCount = descriptors.Count,
            ThreadCount = threadNo
        };

        var taskResults = await Task.WhenAll(descriptors
            .Batches(descriptors.Count / threadNo)
            .Select(x => Task.Run(() =>
            {
                var tmp = x.Select(y => Compute(testedImage, y, context)).ToList();
                context.ThreadCount--;
                return tmp;
            })).ToList());

        var results = taskResults.SelectMany(x => x).ToList();
        results.Sort();

        var resultPath = $"C:\\Projects\\watches\\{testedImage.Name}.txt";
        using (StreamWriter outputFile = new StreamWriter(resultPath))
        {
            Console.WriteLine();
            foreach (var r in results)
            {
                //Console.WriteLine($"{r.Image} {r.Score}");
                outputFile.WriteLine($"{r.Image} {r.Score}");
            }
        }
    }

    private static ComparisionResult Compute(Fature testedImage, Fature y, ParallelContext context)
    {
        var tmp = new ComparisionResult(y.Similarity(testedImage), y.Name);

        lock (context)
        {
            Console.Write($"\rCompute {++context.FinishedCount}\\{context.TotalCount}   Threads: {context.ThreadCount}");
        }

        return tmp;
    }
}

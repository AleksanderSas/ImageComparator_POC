// See https://aka.ms/new-console-template for more information
using Emgu.CV;
using ImageComparatorPOC;
using System.Diagnostics;


//RunTest();
await SearchInDirectory();

static async Task SearchInDirectory()
{
    //Patek Philippe
    string directory = "C:\\Projects\\watches\\SWE-production\\Vacheron Constantin";
    string[] files = Directory.GetFiles(directory);

    //Perek_philippe
    string teseed2 = "C:\\Projects\\watches\\SWE-production\\Stolen\\Vacheron_constatin_Overseas.jpg";
    var testedDescriptor2 = Feature.GetFature(CvInvoke.Imread(teseed2), "test.jpg");

    //For quick test
    //files = files.Take(40).ToArray();

    var timer = new Stopwatch();
    timer.Start();
    var readImageContext = new ParallelContext { TotalCount = files.Length };
    var batches = files.ToList().Batches(files.Length / 10);
    var taskResults = await Task.WhenAll(batches.Select(x => GetFeatureAsync(x, readImageContext)).ToList());

    List<Feature> descriptors = taskResults
        .SelectMany(x => x)
        .Where(x => x != null)
        .ToList();

    Console.WriteLine($"\nRead time: {timer.ElapsedMilliseconds / 1000}");
    Console.WriteLine();

    timer.Restart();
    await Tester.TestAsync(descriptors, testedDescriptor2);
    timer.Stop();
    Console.WriteLine($"\nProcess time: {timer.ElapsedMilliseconds / 1000}");
}

static Task<List<Feature>> GetFeatureAsync(IList<string> files, ParallelContext context)
{
    return Task.Run(() => files.Select(y =>
    {
        var tmp = Feature.GetFature(CvInvoke.Imread(y), y);
        lock(context)
        {
            Console.Write($"\rRead images {++context.FinishedCount}\\{context.TotalCount}");
        }
        return tmp;
    }).ToList());
}
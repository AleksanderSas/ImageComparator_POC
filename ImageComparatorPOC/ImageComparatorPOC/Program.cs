// See https://aka.ms/new-console-template for more information
using Emgu.CV;
using ImageComparatorPOC;
using System.Diagnostics;


//RunTest();
await SearchInDirectory();

static async Task SearchInDirectory()
{
    string directory = "C:\\Projects\\watches\\SWE-production\\Patek Philippe";
    string[] files = Directory.GetFiles(directory);

    string teseed2 = "C:\\Projects\\watches\\SWE-production\\Stolen\\Perek_philippe.jpg";
    var testedDescriptor2 = Fature.GetFature(CvInvoke.Imread(teseed2), "test.jpg");

    //For quick test
    files = files.Take(40).ToArray();

    var timer = new Stopwatch();
    timer.Start();
    var readImageContext = new ParallelContext { TotalCount = files.Length };
    var batches = files.ToList().Batches(files.Length / 10);
    var taskResults = await Task.WhenAll(batches.Select(x => GetFeatureAsync(x, readImageContext)).ToList());

    List<Fature> descriptors = taskResults
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

static Task<List<Fature>> GetFeatureAsync(IList<string> files, ParallelContext context)
{
    return Task.Run(() => files.Select(y =>
    {
        var tmp = Fature.GetFature(CvInvoke.Imread(y), y);
        lock(context)
        {
            Console.Write($"\rRead images {++context.FinishedCount}\\{context.TotalCount}");
        }
        return tmp;
    }).ToList());
}
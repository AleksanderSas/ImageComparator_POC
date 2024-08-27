using Emgu.CV;
using static ImageComparatorPOC.TestRunner;

namespace ImageComparatorPOC.Postprocessing;

public class PostProcessor
{
    private const float Treshold = -115;
    private const string ResultDirectory = "C:\\Projects\\watches\\comparisionResults-S-V2";
    private const string ZyteDir = "C:\\Projects\\watches\\Zyte-production";
    private const string SweDir = "C:\\Projects\\watches\\SWE-production-S";

    public static async Task Process(string root)
    {
        var brands = new DirectoryInfo(root).GetDirectories();
        foreach (var brand in brands)
        {
            await Process(brand);
            Console.WriteLine("completed: " + brand.Name);
        }
    }

    public static async Task Process(DirectoryInfo brand)
    {
        var models = new DirectoryInfo(brand.FullName).GetDirectories();
        var modelBatches = models.Batches(models.Length / 1);
        await Task.WhenAll(modelBatches.Select(batch => Task.Run(async () =>
        {
            foreach (var model in batch)
            {
                var reportFiles = new DirectoryInfo(model.FullName).GetFiles("*.txt");
                foreach (var x in reportFiles)
                {
                    await Process(brand, model, x);
                }
            }
        })).ToList());
    }

    private static async Task Process(DirectoryInfo brand, DirectoryInfo model, FileInfo x)
    {
        try
        {
            var report = await Parse(brand.Name, model.Name, x.Name, x.FullName);
            if (report == null)
            {
                return;
            }
            var jpgName = x.Name.Substring(0, x.Name.Length - 4);
            var sweImageUrl = $"{SweDir}/{brand.Name}/{model.Name}/{jpgName}";
            var resultFile = $"{ResultDirectory}/{brand.Name}/{model.Name}/{x.Name}";
            var resultFileDir = $"{ResultDirectory}/{brand.Name}/{model.Name}";

            if (File.Exists(resultFile) && report.Version2)
            {
                Console.WriteLine($"{jpgName} already processed");
                return;
            }
            var sweImage = Feature.GetFature(CvInvoke.Imread(sweImageUrl), jpgName);

            if(report.Entries == null)
            {
                return;
            }
            var featrures = new FilesToProcess { Files = report.Entries.Where(x => x.Score > Treshold).Select(e => $"{ZyteDir}/{e.OriginImage}").ToList() };
            await new TestRunner().RunTestWithCache(true, resultFileDir, resultFile, sweImageUrl, sweImage, featrures);
        }
        catch (Exception ex)
        {
            Console.WriteLine(ex);
        }
    }

    private static async Task<Report> Parse(string brand, string model, string watch, string reportFile)
    {
        try
        {
            var lines = await File.ReadAllLinesAsync(reportFile);
            var report = new Report
            {
                Version2 = bool.Parse(lines[0].Substring(9)),
                Comparisions = Int32.Parse(lines[2].Split(" ")[1]),
                Found = false,
                Brand = brand,
                Model = model,
                Image = watch
            };

            if (lines.Length > 5)
            {
                report.Entries = lines.Skip(5).Select(x =>
                {
                    return BuildEntry(brand, model, watch, x);
                }).ToList();
                
                report.Found = true;
            }
            return report;
        }
        catch(Exception ex) 
        {
            Console.WriteLine(ex);
            return null;
        }
    }

    private static ReportEntry BuildEntry(string brand, string model, string watch, string x)
    {
        var match = x.Split(';');
        var dotPosition = match[0].LastIndexOf('.');
        var slashPosition = Math.Max(match[0].LastIndexOf('\\'), match[0].LastIndexOf('/'));
        int slashIdx = match[0].LastIndexOf('\\');
        slashIdx = match[0].LastIndexOf('\\', slashIdx - 1);
        slashIdx = match[0].LastIndexOf('\\', slashIdx - 1);
        return new ReportEntry
        {
            Score = double.Parse(match[1]),
            ScorePerPoint = double.Parse(match[2]),
            ListingId = long.Parse(match[0].Substring(slashPosition + 1, dotPosition - slashPosition - 1)),
            Brand = brand,
            Model = model,
            Image = watch.Substring(0, watch.Length - 4),
            OriginImage = match[0].Substring(slashIdx + 1)
        };
    }

    public class Report
    {
        public bool Version2;
        public string Model;
        public string Brand;
        public string Image;
        public bool Found;
        public int Comparisions;
        public List<ReportEntry> Entries;
    }

    public class ReportEntry
    {
        public string OriginImage;
        public string Model;
        public string Brand;
        public string Image;

        public long ListingId;
        public double Score;
        public double ScorePerPoint;
    }
}

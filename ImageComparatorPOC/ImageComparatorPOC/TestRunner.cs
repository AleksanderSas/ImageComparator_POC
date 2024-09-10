using Emgu.CV;
using Emgu.CV.Structure;
using System.Collections.Concurrent;
using System.Drawing;

namespace ImageComparatorPOC
{
    internal class TestRunner
    {
        private ConcurrentDictionary<string, Feature> _features = new ConcurrentDictionary<string, Feature>();

        private const string ZyteFeatureDir = "C:\\Projects\\watches\\Zyte-features\\";
        private const string ResultsDir = "C:\\Projects\\watches\\comparisionResults-S\\";
        private const string ResultsDir2 = "C:\\Projects\\watches\\comparisionResults\\SWE_SINGLE\\";
        private const string ZyteDir = "C:\\Projects\\watches\\Zyte-production\\";
        private const string SweDir = "C:\\Projects\\watches\\SWE-production-S\\";

        public async Task ScanSwe(string image1, string image2, string brand, string model, bool useGeometryFeature)
        {
            var dir = $"{SweDir}{brand}\\{model}";
            var models = new DirectoryInfo(dir).GetFiles();
            List<Feature> descriptors = await GetFeature(models.Select(x => x.FullName).ToList(), false, useGeometryFeature);
            await ScanSwe2(image1, brand, model, descriptors);
            await ScanSwe2(image2, brand, model, descriptors);
        }

        public static async Task PrintPointMatching(string f1, string f2)
        {
            var input1 = Feature.GetFature(CvInvoke.Imread(f1), "f1");
            var input2 = Feature.GetFature(CvInvoke.Imread(f2), "f2");

            var mapping = input1.GetPointMapping(input2, 300);
            Mat output = new Mat
            (
                Math.Max(input1.ResisedImage.Rows, input2.ResisedImage.Rows),
                input1.ResisedImage.Cols + input2.ResisedImage.Cols,
                input1.ResisedImage.Depth,
                input1.ResisedImage.NumberOfChannels
            );



            input1.CopyInto(output, 0);
            input2.CopyInto(output, input1.ResisedImage.Cols);

            for(int i = 0; i < 30; i++)
            {
                DrawLine(input1, input2, mapping, output, i, new MCvScalar(255, 0, 0));
            }

            for (int i = 220; i < 250; i++)
            {
                DrawLine(input1, input2, mapping, output, i, new MCvScalar(0, 0, 255));
            }

            CvInvoke.Imwrite("C:\\Projects\\watches\\composition.png", output);
        }

        private static void DrawLine(Feature input1, Feature input2, List<(double score, int matchIdx, int originIdx)> mapping, Mat output, int i, MCvScalar color)
        {
            var idx = mapping[i].matchIdx;
            var data = input2.Descriptor.GetData();
            var p1 = new Point((int)(float)data.GetValue(idx, 64) + input1.ResisedImage.Cols, (int)(float)data.GetValue(idx, 65));

            idx = mapping[i].originIdx;
            data = input1.Descriptor.GetData();
            var p2 = new Point((int)(float)data.GetValue(idx, 64), (int)(float)data.GetValue(idx, 65));

            CvInvoke.Line(output, p1, p2, color);
        }

        public async Task RunTestWithCache(string brand, bool useGeometryFeature)
        {
            var dir = $"{SweDir}{brand}";
            var models = new DirectoryInfo(dir).GetDirectories();
            foreach (var model in models)
            {
                await RunTestWithCache(brand, model.Name, useGeometryFeature);
            }
        }

        public async Task RunTestWithCache(string brand, string model, bool useGeometryFeature)
        {
            var dir = $"{SweDir}{brand}\\{model}";
            var files = new DirectoryInfo(dir).GetFiles();
            foreach (var file in files)
            {
                try
                {
                    Console.WriteLine(file.Name);
                    await RunTestWithCache(file.Name, brand, model, useGeometryFeature);
                }
                catch (Exception ex) 
                {
                    Console.WriteLine($"\nERROR: {file.Name} - {ex}");
                }
            }
        }

        public async Task ScanSwe2(string filename, string brand, string model, List<Feature> descriptors)
        {
            var resultDir = $"{ResultsDir2}{brand}\\{model}";

            var idx = Math.Max(filename.LastIndexOf('/'), filename.LastIndexOf('\\'));
            var name = filename.Substring(idx + 1); 
            var watch = Feature.GetFature(CvInvoke.Imread(filename), name);
            var resultPath = $"{resultDir}\\{name}.txt";

            //For test purpose
            //filesNames.Files = filesNames.Files.Skip(400).ToList();
            var results = await Tester.TestAsync(descriptors, watch, true);

            if (!Directory.Exists(resultDir))
            {
                Directory.CreateDirectory(resultDir);
            }
            using (StreamWriter outputFile = new StreamWriter(resultPath))
            {
                outputFile.WriteLine($"File: {filename}");
                outputFile.WriteLine($"Scan_no: {descriptors.Count}");
                outputFile.WriteLine($"Brands: {brand}");
                outputFile.WriteLine($"Models: {model}");
                foreach (var r in results.Where(x => x.Score < 0.0))// && x.Score > -180.0))
                {
                    outputFile.WriteLine($"{r.Image};{r.Score};{r.ScorePerPoint}");
                }

            }
        }

        public async Task RunTestWithCache(string filename, string brand, string model, bool useGeometryFeature)
        {
            var resultDir = $"{ResultsDir}{brand}\\{model}";
            var resultPath = $"{resultDir}\\{filename}.txt";
            if (File.Exists(resultPath))
            {
                Console.WriteLine("Results already exists");
                return;
            }

            var watchFile = $"{SweDir}{brand}\\{model}\\{filename}";
            var watch = Feature.GetFature(CvInvoke.Imread(watchFile), filename);

            var filesNames = GetFilesToCompare(brand, model);
            //For test purpose
            //filesNames.Files = filesNames.Files.Skip(400).ToList();
            await RunTestWithCache(useGeometryFeature, resultDir, resultPath, watchFile, watch, filesNames);
        }

        public async Task RunTestWithCache(bool useGeometryFeature, string resultDir, string resultPath, string watchFile, Feature watch, FilesToProcess filesNames, int pointsLimit = 200)
        {
            var descriptors = await GetFeature(filesNames.Files, true, useGeometryFeature);
            var results = await Tester.TestAsync(descriptors, watch, useGeometryFeature, pointsLimit);

            SaveResult(useGeometryFeature, resultDir, resultPath, watchFile, filesNames, descriptors, results);
        }

        public async Task RunPostProcessWithCache(bool useGeometryFeature, string resultDir, string resultPath, string watchFile, Feature watch, FilesToProcess filesNames, int pointsLimit)
        {
            var descriptors = await GetFeature(filesNames.Files, true, useGeometryFeature);
            var results = await Tester.TestInternalAsync(descriptors, watch, 300, pointsLimit, useGeometryFeature);

            SaveResult(useGeometryFeature, resultDir, resultPath, watchFile, filesNames, descriptors, results);
        }

        private static void SaveResult(bool useGeometryFeature, string resultDir, string resultPath, string watchFile, FilesToProcess filesNames, List<Feature> descriptors, List<ComparisionResult> results)
        {
            if (!Directory.Exists(resultDir))
            {
                Directory.CreateDirectory(resultDir);
            }
            using (StreamWriter outputFile = new StreamWriter(resultPath))
            {
                var version = useGeometryFeature ? 1 : 0;
                outputFile.WriteLine($"Version: {useGeometryFeature}");
                outputFile.WriteLine($"File: {watchFile}");
                outputFile.WriteLine($"Scan_no: {descriptors.Count}");
                outputFile.WriteLine($"Brands: {string.Join(";", filesNames.Brands)}");
                outputFile.WriteLine($"Models: {string.Join(";", filesNames.Models)}");
                var scoreLimit = useGeometryFeature ? -280.0 : -180.0;
                foreach (var r in results.Where(x => x.Score < 0.0 && x.Score > scoreLimit))
                {
                    outputFile.WriteLine($"{r.Image};{r.Score};{r.ScorePerPoint}");
                }

            }
        }

        private FilesToProcess GetFilesToCompare(string brand, string model)
        {
            var result = new FilesToProcess();
            var dirs = new DirectoryInfo(ZyteDir).GetDirectories();
            var brandDirs = dirs.Where(x => IsNameSimilar(brand, x.Name)).ToList();
            foreach (var brandDir in brandDirs)
            {
                result.Brands.Add(brandDir.Name);
                dirs = new DirectoryInfo(brandDir.FullName).GetDirectories();
                var modelDirs = dirs.Where(x => IsNameSimilar(model, x.Name)).ToList();

                foreach (var modelDir in modelDirs)
                {
                    result.Models.Add(modelDir.Name);
                    var files = new DirectoryInfo(modelDir.FullName).GetFiles();
                    result.Files.AddRange(files.Select(x => x.FullName));
                }
            }
            return result;
        }

        private async Task<List<Feature>> GetFeature(List<string> filenames, bool saveFeatures, bool useGeometryFeature)
        {
            var readImageContext = new ParallelContext { TotalCount = filenames.Count };
            var batches = filenames.ToList().Batches(filenames.Count / 6);
            //var batches = filenames.ToList().Batches(filenames.Count);
            var taskResults = await Task.WhenAll(batches.Select(x => GetFeatureAsync(x, readImageContext, saveFeatures, useGeometryFeature)).ToList());
            Console.WriteLine();

            List<Feature> descriptors = taskResults
                .SelectMany(x => x)
                .Where(x => x != null)
                .Cast<Feature>()
                .ToList();

            return descriptors;
        }

        Task<List<Feature?>> GetFeatureAsync(IList<string> files, ParallelContext context, bool saveFeatures, bool useGeometryFeature)
        {
            return Task.Run(() => files.Select(y =>
            {
                try
                {
                    if (!_features.TryGetValue(y, out var feature))
                    {
                        feature = ReadFeature(y, saveFeatures, useGeometryFeature);
                    }

                    var c = Interlocked.Increment(ref context.FinishedCount);
                    if(c % 8 == 0)
                    {
                        lock (context)
                        {
                            Console.Write($"\rRead images {c}\\{context.TotalCount} MPI: {context.MilisPerImage()}");

                        }
                    }
                    return feature;
                }
                catch(Exception ex) 
                {
                    return null;
                }
            }).ToList());
        }

        private Feature ReadFeature(string filename, bool saveFeatures, bool useGeometryFeature)
        {
            var cachFile = filename.Replace(ZyteDir, ZyteFeatureDir) + ".PFM";
            var feature = Feature.Load(cachFile, filename, useGeometryFeature);
            if(feature == null)
            {
                feature = Feature.GetFature(CvInvoke.Imread(filename), filename);
                if (saveFeatures && feature != null)
                {
                    feature.Save(cachFile);
                }
            }

                _features[filename] = feature;

            return feature;
        }

        private bool IsNameSimilar(string swe, string zyte)
        {
            swe = swe.Replace("'", "").Replace("\"", " ").Replace("&", " ").Replace("-", " ");
            var weWords = swe.Split(' ').Where(x => x.Length > 1).ToList();

            zyte = zyte.Replace("'", "").Replace("\"", " ").Replace("&", " ").Replace("-", " ");
            var zyteWords = zyte.Split(' ').Where(x => x.Length > 1).ToList();

            if(weWords.Count == 0|| zyteWords.Count == 0)
            {
                //Console.WriteLine($"'{s1}' '{s2}' fallback");
                return true;
            }

            return weWords.All(zyteWords.Contains);
        }

        public struct FilesToProcess
        {
            public List<string> Files;
            public List<string> Brands;
            public List<string> Models;

            public FilesToProcess()
            {
                Files = new List<string>();
                Brands= new List<string>();
                Models= new List<string>();
            }
        }
    }
}

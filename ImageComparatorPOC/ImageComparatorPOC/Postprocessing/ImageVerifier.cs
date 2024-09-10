using Emgu.CV;
using OfficeOpenXml;
using System.Drawing;
using System.Net;

namespace ImageComparatorPOC.Postprocessing
{
    public class ImageVerifier
    {
        private string[] Lines;
        private List<(string zyteImage, string sweImage)> Lines2 = new List<(string, string)>();
        string reportFile;

        public ImageVerifier(string reportFile) 
        {
            this.reportFile = reportFile;
            

            //Lines = File.ReadAllLines(reportFile);
        }

        public async Task xxx()
        {
            using var webClient = new WebClient();
            using (var package = new ExcelPackage(new FileInfo(reportFile)))
            {
                ExcelWorksheet worksheet = package.Workbook.Worksheets[1];
                int rowCount = worksheet.Dimension.End.Row;     //get row count
                Task<(string, Mat)> next = null;
                for (int row = 120; row <= rowCount; row++)
                {
                    string fileName = null;
                    Mat m = null;
                    if (next == null)
                    {
                        (fileName, m) = await Read(webClient, worksheet, row);
                    }
                    else
                    {
                        (fileName, m) = await next;
                    }

                    if (row + 1 <= rowCount)
                    {
                        var nextSweImage = worksheet.Cells[row + 1, 7].Value.ToString().Trim();
                        var nextZyteImage = worksheet.Cells[row + 1, 9].Value.ToString().Trim();
                        next = ReadImages(nextSweImage, nextZyteImage, webClient, row + 1);
                    }
                    else
                    {
                        package.Save();
                        return;
                    }

                    CvInvoke.Imshow(fileName, m);
                    var pressedButton = ReadButton();
                    switch (pressedButton)
                    {
                        case 13: //enter
                            worksheet.Row(row).Style.Fill.PatternType = OfficeOpenXml.Style.ExcelFillStyle.Solid;
                            worksheet.Row(row).Style.Fill.BackgroundColor.SetColor(Color.LightGreen);
                            break;
                        case 32: //space
                            worksheet.Row(row).Style.Fill.PatternType = OfficeOpenXml.Style.ExcelFillStyle.Solid;
                            worksheet.Row(row).Style.Fill.BackgroundColor.SetColor(Color.LightYellow);
                            break;
                        case 9:  //tab
                            worksheet.Row(row).Style.Fill.PatternType = OfficeOpenXml.Style.ExcelFillStyle.Solid;
                            worksheet.Row(row).Style.Fill.BackgroundColor.SetColor(Color.LightGray);
                            break;
                        case 27: //esc
                            package.Save();
                            return;
                    }
                    CvInvoke.DestroyWindow(fileName);
                }
            }
        }

        private async Task<(string fileName, Mat m)> Read(WebClient webClient, ExcelWorksheet worksheet, int row)
        {
            var sweImage = worksheet.Cells[row, 7].Value.ToString().Trim();
            var zyteImage = worksheet.Cells[row, 9].Value.ToString().Trim();

            return await ReadImages(sweImage, zyteImage, webClient, row);
        }

        public async Task RunVerifier()
        {

            int lineNo = 33;
            using var webClient = new WebClient();

            var (fileName, m) = await ReadImages(lineNo, webClient, Lines[lineNo-1]);

            foreach (var line in Lines.Skip(lineNo))
            {
                var t = ReadImages(lineNo, webClient, line);
                //var imageViewer = "C:\\ProgramData\\Microsoft\\Windows\\Start Menu\\Programs\\Accessories\\Paint.lnk ";
                //var process = Process.Start("mspaint", fileName);
                //var process = Process.Start("explorer.exe", fileName);

                CvInvoke.Imshow(fileName, m);
                var pressedButton = ReadButton();
                switch (pressedButton)
                {
                    case 13: //enter
                    case 32: //space
                    case 9:  //tab
                        break;
                    case 27: //esc
                        return;
                }
                CvInvoke.DestroyWindow(fileName);

                lineNo++;

                //File.Delete(fileName);

                (fileName, m) = await t;
            }
        }

        public async Task RunVerifier(string sweImage, string zyteImage, int lineNo, WebClient webClient)
        {

            var (fileName, m) = await ReadImages(sweImage, zyteImage, webClient, lineNo);

            CvInvoke.Imshow(fileName, m);
            var pressedButton = ReadButton();
            switch (pressedButton)
            {
                case 13: //enter
                case 32: //space
                case 9:  //tab
                    break;
                case 27: //esc
                    return;
            }
            CvInvoke.DestroyWindow(fileName);
        }

        private static int ReadButton()
        {
            while (true)
            {
                var pressedButton = CvInvoke.WaitKey();

                switch (pressedButton)
                {
                    case 13: //enter
                    case 32: //space
                    case 27: //esc
                    case 9: //tab
                        return pressedButton;
                    default:
                        Console.WriteLine("Press:\n * ENTER to accept\n * SPACE to make as suspicious\n * TAB to reject\n * ECS to exit");
                        break;
                }
            }
        }

        private static async Task<(string, Mat)> ReadImages(int lineNo, WebClient webClient, string? line)
        {
            var data = line.Split(';');
            var sweImg = data[6];
            var zyteImg = data[8];
            await webClient.DownloadFileTaskAsync(sweImg, "swe.jpg");
            await webClient.DownloadFileTaskAsync(zyteImg, "zyte.jpg");

            var sweTmp = new Mat("swe.jpg");
            Mat swe = new Mat();

            if (sweTmp.Width > 1000)
            {
                CvInvoke.Resize(sweTmp, swe, new Size(900, (int)(900.0 / sweTmp.Width * sweTmp.Height)));
            }
            else
            {
                swe = sweTmp;
            }

            var zyte = new Mat("zyte.jpg");

            Mat output = new Mat
            (
               Math.Max(swe.Rows, zyte.Rows),
               swe.Cols + zyte.Cols,
               zyte.Depth,
               zyte.NumberOfChannels
            );

            Copy(output, swe, 0);
            Copy(output, zyte, swe.Cols);

            var fileName = $"{lineNo}________{data[7]}.jpg";
            //output.Save(fileName);
            return (fileName, output);
        }

        private async Task<(string, Mat)> ReadImages(string sweImg, string zyteImg, WebClient webClient, int lineNo)
        {
            await webClient.DownloadFileTaskAsync(sweImg, "swe.jpg");
            await webClient.DownloadFileTaskAsync(zyteImg, "zyte.jpg");

            var sweTmp = new Mat("swe.jpg");
            Mat swe = new Mat();

            if (sweTmp.Width > 1000)
            {
                CvInvoke.Resize(sweTmp, swe, new Size(900, (int)(900.0 / sweTmp.Width * sweTmp.Height)));
            }
            else
            {
                swe = sweTmp;
            }


            var zyteTmp = new Mat("zyte.jpg");
            var zyte = new Mat();
            if (zyteTmp.Width > 1000)
            {
                CvInvoke.Resize(zyteTmp, zyte, new Size(900, (int)(900.0 / zyteTmp.Width * zyteTmp.Height)));
            }
            else
            {
                zyte = zyteTmp;
            }

            Mat output = new Mat
            (
               Math.Max(swe.Rows, zyte.Rows),
               swe.Cols + zyte.Cols,
               zyte.Depth,
               zyte.NumberOfChannels
            );

            Copy(output, swe, 0);
            Copy(output, zyte, swe.Cols);

            var fileName = $"{lineNo}";
            return (fileName, output);
        }

        public static unsafe void Copy(Mat dst, Mat mat, int xShift)
        {
            var cols = mat.Cols;
            for (int y = 0; y < mat.Rows; y++)
            {
                var source = mat.Row(y);
                var destination = dst.Row(y);

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
}

using Emgu.CV.Features2D;

namespace ImageComparatorPOC;

internal class ComparisionResult : IComparable<ComparisionResult>
{
    public ComparisionResult(double score, double scorePerPoint, string image, Feature otherFeature, double diffScore, double angleScore)
    {
        Score = score;
        ScorePerPoint = scorePerPoint;
        Image = image;
        OtherFeature = otherFeature;
        DiffScore = diffScore;
        AngleScore = angleScore;
     }

    public double Score { get; set; }

    public double DiffScore { get; set; }

    public double AngleScore { get; set; }

    public double ScorePerPoint { get; set; }

    public string Image {  get; set; }

    public Feature OtherFeature { get; set; }

    public int CompareTo(ComparisionResult? other) => other.Score.CompareTo(Score);
}

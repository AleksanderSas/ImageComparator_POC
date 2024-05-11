using Emgu.CV.Features2D;

namespace ImageComparatorPOC;

internal class ComparisionResult : IComparable<ComparisionResult>
{
    public ComparisionResult(double score, double scorePerPoint, string image, Feature otherFeature)
    {
        Score = score;
        ScorePerPoint = scorePerPoint;
        Image = image;
        OtherFeature = otherFeature;
    }

    public double Score { get; set; }

    public double ScorePerPoint { get; set; }

    public string Image {  get; set; }

    public Feature OtherFeature { get; set; }

    public int CompareTo(ComparisionResult? other) => other.Score.CompareTo(Score);
}

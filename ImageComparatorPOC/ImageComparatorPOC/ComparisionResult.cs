using Emgu.CV.Features2D;

namespace ImageComparatorPOC;

internal class ComparisionResult : IComparable<ComparisionResult>
{
    public ComparisionResult(double score, string image, Feature otherFeature)
    {
        Score = score;
        Image = image;
        OtherFeature = otherFeature;
    }

    public double Score { get; set; }

    public string Image {  get; set; }

    public Feature OtherFeature { get; set; }

    public int CompareTo(ComparisionResult? other) => other.Score.CompareTo(Score);
}

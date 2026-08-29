package ru.traphouse.functions.log;

public class Log5 {
    private final Ln ln;
    private final double LN5;

    public Log5(Ln ln) {
        this.ln = ln;
        this.LN5 = ln.calculate(5.0);
    }

    // log₅(x) = ln(x) / ln(5)
    public double calculate(double x) {
        return ln.calculate(x) / LN5;
    }
}

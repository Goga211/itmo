package ru.traphouse.functions.trig;

public class Cos {
    private final Sin sin;

    public Cos(Sin sin) {
        this.sin = sin;
    }

    // cos(x) = sin(π/2 - x)
    public double calculate(double x) {
        return sin.calculate(Math.PI / 2 - x);
    }
}

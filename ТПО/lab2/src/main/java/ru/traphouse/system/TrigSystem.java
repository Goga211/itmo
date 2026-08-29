package ru.traphouse.system;

import ru.traphouse.functions.trig.Cot;
import ru.traphouse.functions.trig.Csc;
import ru.traphouse.functions.trig.Sin;

/**
 * Тригонометрическая часть системы (x <= 0):
 * (((((cot(x) / csc(x)) / sin(x)) - csc(x)) ^ 3) ^ 2)
 */
public class TrigSystem {
    private final Sin sin;
    private final Cot cot;
    private final Csc csc;

    public TrigSystem(Sin sin, Cot cot, Csc csc) {
        this.sin = sin;
        this.cot = cot;
        this.csc = csc;
    }

    public double calculate(double x) {
        // cot и csc сами бросят ArithmeticException если sin(x) = 0
        double cotVal = cot.calculate(x);
        double cscVal = csc.calculate(x);
        double sinVal = sin.calculate(x);

        double inner = ((cotVal / cscVal) / sinVal) - cscVal;
        double pow3 = inner * inner * inner;
        return pow3 * pow3;
    }
}

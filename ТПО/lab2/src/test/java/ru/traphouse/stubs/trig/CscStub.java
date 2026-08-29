package ru.traphouse.stubs.trig;

import ru.traphouse.functions.trig.Csc;
import ru.traphouse.functions.trig.Sin;

import java.util.HashMap;
import java.util.Map;

public class CscStub extends Csc {
    private static final Map<Double, Double> TABLE = new HashMap<>();

    static {
        TABLE.put(Math.PI / 2,   1.0);
        TABLE.put(-Math.PI / 2, -1.0);
        TABLE.put(-1.0,         -1.1883951057781212);
        TABLE.put(-2.0,         -1.0997501702946164);
        TABLE.put(-3.0,         -7.086167395737498);
    }

    public CscStub(Sin sin) {
        super(sin);
    }

    @Override
    public double calculate(double x) {
        for (Map.Entry<Double, Double> entry : TABLE.entrySet()) {
            if (Math.abs(entry.getKey() - x) < 1e-9) {
                return entry.getValue();
            }
        }
        return super.calculate(x);
    }
}

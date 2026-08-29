package ru.traphouse.stubs.log;

import ru.traphouse.functions.log.Log5;

/**
 * Заглушка Log5 с фиксированным возвращаемым значением.
 */
public class FixedLog5 extends Log5 {
    private final double fixedValue;

    public FixedLog5(double fixedValue) {
        super(new FixedLn(1.0));
        this.fixedValue = fixedValue;
    }

    @Override
    public double calculate(double x) {
        return fixedValue;
    }
}

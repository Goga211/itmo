package ru.traphouse.system;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import ru.traphouse.functions.log.*;
import ru.traphouse.functions.trig.*;

import static org.junit.jupiter.api.Assertions.*;

class FunctionSystemTest {
    private static final double EPS = 1e-4;
    private FunctionSystem system;

    @BeforeEach
    void setUp() {
        Sin sin = new Sin(1e-10);
        Cos cos = new Cos(sin);
        Cot cot = new Cot(sin, cos);
        Csc csc = new Csc(sin);
        TrigSystem trigSystem = new TrigSystem(sin, cot, csc);

        Ln    ln    = new Ln(1e-10);
        Log2  log2  = new Log2(ln);
        Log3  log3  = new Log3(ln);
        Log5  log5  = new Log5(ln);
        Log10 log10 = new Log10(ln);
        LogSystem logSystem = new LogSystem(ln, log2, log3, log5, log10);

        system = new FunctionSystem(trigSystem, logSystem);
    }

    @Test
    void testNegativeXUsesTrig() {
        double x = -1.0;
        double expected = Math.pow((Math.cos(x) - 1.0) / Math.sin(x), 6);
        assertEquals(expected, system.calculate(x), EPS);
    }

    @Test
    void testPositiveXUsesLog() {
        assertDoesNotThrow(() -> system.calculate(2.0));
    }

    @Test
    void testBoundaryZeroIsTrig() {
        assertThrows(ArithmeticException.class, () -> system.calculate(0.0));
    }

    @Test
    void testXEqualsOneIsUndefined() {
        assertThrows(ArithmeticException.class, () -> system.calculate(1.0));
    }
}

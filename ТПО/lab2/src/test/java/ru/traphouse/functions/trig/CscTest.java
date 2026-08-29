package ru.traphouse.functions.trig;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CscTest {
    private static final double EPS = 1e-6;
    private Csc csc;

    @BeforeEach
    void setUp() {
        Sin sin = new Sin(1e-10);
        csc = new Csc(sin);
    }

    @Test
    void testPiHalf() {
        assertEquals(1.0, csc.calculate(Math.PI / 2), EPS);
    }

    @Test
    void testMinusPiHalf() {
        assertEquals(-1.0, csc.calculate(-Math.PI / 2), EPS);
    }

    @Test
    void testKnownValue() {
        assertEquals(1.0 / Math.sin(-1.0), csc.calculate(-1.0), EPS);
    }

    @Test
    void testUndefinedAtZero() {
        assertThrows(ArithmeticException.class, () -> csc.calculate(0.0));
    }

    @Test
    void testUndefinedAtPi() {
        assertThrows(ArithmeticException.class, () -> csc.calculate(Math.PI));
    }
}

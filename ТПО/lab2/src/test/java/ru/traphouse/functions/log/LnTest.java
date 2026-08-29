package ru.traphouse.functions.log;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import ru.traphouse.stubs.log.LnStub;

import static org.junit.jupiter.api.Assertions.*;

class LnTest {
    private static final double EPS = 1e-6;
    private Ln ln;

    @BeforeEach
    void setUp() {
        ln = new Ln(1e-10);
    }

    @Test
    void testOne() {
        assertEquals(0.0, ln.calculate(1.0), EPS);
    }

    @Test
    void testE() {
        assertEquals(1.0, ln.calculate(Math.E), EPS);
    }

    @Test
    void testTwo() {
        assertEquals(Math.log(2), ln.calculate(2.0), EPS);
    }

    @Test
    void testFraction() {
        assertEquals(Math.log(0.5), ln.calculate(0.5), EPS);
    }

    @Test
    void testUndefinedZero() {
        assertThrows(ArithmeticException.class, () -> ln.calculate(0.0));
    }

    @Test
    void testUndefinedNegative() {
        assertThrows(ArithmeticException.class, () -> ln.calculate(-1.0));
    }

    @Test
    void testStubMatchesReal() {
        LnStub stub = new LnStub();
        assertEquals(stub.calculate(2.0), ln.calculate(2.0), EPS);
        assertEquals(stub.calculate(10.0), ln.calculate(10.0), EPS);
    }
}

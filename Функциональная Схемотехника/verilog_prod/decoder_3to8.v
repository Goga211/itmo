module decoder_3to8 (
    input  wire A, B, C,
    input  wire EN,
    output wire Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7
);
    wire nA, nB, nC;
    wire y0n, y1n, y2n, y3n, y4n, y5n, y6n, y7n;

    nand (nA, A, A);
    nand (nB, B, B);
    nand (nC, C, C);

    nand (y0n, EN, nA, nB, nC); // 000
    nand (y1n, EN, nA, nB,  C); // 001
    nand (y2n, EN, nA,  B, nC); // 010
    nand (y3n, EN, nA,  B,  C); // 011
    nand (y4n, EN,  A, nB, nC); // 100
    nand (y5n, EN,  A, nB,  C); // 101
    nand (y6n, EN,  A,  B, nC); // 110
    nand (y7n, EN,  A,  B,  C); // 111

    nand (Y0, y0n, y0n);
    nand (Y1, y1n, y1n);
    nand (Y2, y2n, y2n);
    nand (Y3, y3n, y3n);
    nand (Y4, y4n, y4n);
    nand (Y5, y5n, y5n);
    nand (Y6, y6n, y6n);
    nand (Y7, y7n, y7n);
endmodule

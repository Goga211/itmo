module func (
  input  wire        clk,
  input  wire        rst,
  input  wire        start,
  input  wire [7:0]  a,
  input  wire [7:0]  b,
  output reg         busy,
  output reg         valid,
  output reg  [23:0] y
);
  reg         m_start;
  wire        m_busy, m_done;
  reg  [15:0] m_A;
  reg  [7:0]  m_B;
  wire [23:0] m_Y;

  mult16x8 mul (
    .clk(clk), .rst(rst),
    .start(m_start),
    .A(m_A), .B(m_B),
    .busy(m_busy), .done(m_done),
    .Y(m_Y)
  );

  reg [7:0]   a_r, b_r;
  reg [23:0]  ab24, a3_24;
  reg [15:0]  a2_16;

  localparam S_IDLE=3'd0, S_AB=3'd1, S_A2=3'd2, S_A3=3'd3, S_ADD=3'd4, S_DONE=3'd5;
  reg [2:0] st;

  always @(posedge clk) begin
    if (rst) begin
      st<=S_IDLE; busy<=0; valid<=0; y<=0;
      m_start<=0; m_A<=0; m_B<=0;
      a_r<=0; b_r<=0; ab24<=0; a2_16<=0; a3_24<=0;
    end else begin
      m_start <= 1'b0;
      valid   <= 1'b0;
      case (st)
        S_IDLE: begin
          busy <= 1'b0;
          if (start) begin
            busy <= 1'b1;
            a_r  <= a; b_r <= b;

            m_A    <= {8'b0, a};
            m_B    <= b;
            m_start<= 1'b1;
            st     <= S_AB;
          end
        end
        S_AB: if (m_done) begin
          ab24 <= m_Y;            

          m_A    <= {8'b0, a_r};
          m_B    <= a_r;
          m_start<= 1'b1;
          st     <= S_A2;
        end
        S_A2: if (m_done) begin
          a2_16 <= m_Y[15:0];      

          m_A    <= {8'b0, m_Y[15:0]}; 
          m_B    <= a_r;
          m_start<= 1'b1;
          st     <= S_A3;
        end
        S_A3: if (m_done) begin
          a3_24 <= m_Y;              
          st    <= S_ADD;
        end
        S_ADD: begin
          y   <= ab24 + a3_24;        
          st  <= S_DONE;
        end
        S_DONE: begin
          valid <= 1'b1;
          busy  <= 1'b0;
          st    <= S_IDLE;
        end
      endcase
    end
  end
endmodule

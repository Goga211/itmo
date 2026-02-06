import React, { useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import { editTable, setLoading } from "../../redux/generalSlice";
import axiosInstance from "../../axiosInstance";
import styled from "styled-components";

const TableContainer = styled.table`
  max-height: 450px;
  overflow-y: auto;
  display: block;
  border: 2px solid grey;
  border-collapse: collapse;
  width: 100%;
  background: white;
  border-radius: 20px;
    

`;

const TableHeader = styled.th`
    width: calc(100% / 6);
  text-align: center;
  background-color: #f4f4f4;
`;

const TableRow = styled.tr`
    width: calc(100% / 6); /* Делим ширину таблицы на количество столбцов */
  &:nth-child(even) {
    background-color: #f9f9f9;
  }
`;

const TableCell = styled.td`
    
    width: 100%;
    word-wrap: break-word;
    text-align: center;
  padding: 10px;
`;

function Table() {
    const table = useSelector((state) => state.tableEditor.table);
    const refreshTable = useSelector((state) => state.tableEditor.refreshTable);
    const loading = useSelector((state) => state.tableEditor.loading);

    const dispatch = useDispatch();

    async function fill_table() {
        dispatch(setLoading(true));
        try {
            const response = await axiosInstance.get("/dot/get-list");
            dispatch(editTable(response.data));
        } catch (e) {
            console.error("Error fetching table data:", e);
        } finally {
            dispatch(setLoading(false));
        }
    }

    useEffect(() => {
        fill_table();
    }, [refreshTable]);

    // Функция преобразования временной метки в локальное время
    const formatTimestampToLocalTime = (timestamp) => {
        const date = new Date(timestamp);
        return date.toLocaleString(); // Преобразует дату в строку с учетом локального часового пояса
    };

    const listDot = table.map((dot, index) => (
        <TableRow key={index}>
            <TableCell>{dot.x}</TableCell>
            <TableCell>{dot.y}</TableCell>
            <TableCell>{dot.r}</TableCell>
            <TableCell>{dot.executionTime}</TableCell>
            <TableCell>{formatTimestampToLocalTime(dot.time)}</TableCell>
            <TableCell>{dot.result ? "True" : "False"}</TableCell>
        </TableRow>
    ));

    return (
        <TableContainer>
            <thead>
            <tr>
                <TableHeader>X</TableHeader>
                <TableHeader>Y</TableHeader>
                <TableHeader>R</TableHeader>
                <TableHeader>Time</TableHeader>
                <TableHeader>Now</TableHeader>
                <TableHeader>Result</TableHeader>
            </tr>
            </thead>
            <tbody>{listDot}</tbody>
        </TableContainer>
    );
}

export default Table;


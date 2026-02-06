import React from "react";

import "./App.css";
import styled from "styled-components";
import {Route, Routes} from "react-router-dom";
import {Login} from "./components/login/login";
import {Home} from "./components/home/main";
import {Register} from "./components/register/register";
import {Provider} from "react-redux";
import store from './redux/store'


const AppContainer = styled.div`
    width: 100%;
    height: 100%;
`;

function App() {
    return (<Provider store={store}>
            <AppContainer>
                <Routes>
                    <Route
                        path="/home"
                        element={
                        <Home/>
                        }
                    ></Route>
                    <Route path="/" element={<Login/>}></Route>
                    <Route path="/login" element={<Login/>}></Route>
                    <Route path="/register" element={<Register/>}></Route>
                </Routes>
            </AppContainer>
        </Provider>
    );
}

export default App;

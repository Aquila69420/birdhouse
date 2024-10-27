import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import './assets/css/App.css';
import { Provider } from 'react-redux';
import {store, persistor} from './redux/store.js'

import App from './App';
import { PersistGate } from 'redux-persist/integration/react'; // Import PersistGate


const root = ReactDOM.createRoot(document.getElementById('root'));

root.render(
  <Provider store={store}>
    <PersistGate loading={null} persistor={persistor}>
    <BrowserRouter>
    <App />
    </BrowserRouter>
    </PersistGate>
  </Provider>
);

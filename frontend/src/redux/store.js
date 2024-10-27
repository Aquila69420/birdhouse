import { configureStore } from '@reduxjs/toolkit';
import { persistStore, persistReducer } from 'redux-persist';
import storage from 'redux-persist/lib/storage'; // defaults to localStorage for web
import personReducer from './reducers/nameReducer.js';

const persistConfig = {
  key: 'root', // key for storage
  storage, // storage engine to use (localStorage in this case)
};

const persistedReducer = persistReducer(persistConfig, personReducer);

export const store = configureStore({
  reducer: {
    person: persistedReducer,
  },
});

// Create a persistor for use with PersistGate
export const persistor = persistStore(store);

export default store;

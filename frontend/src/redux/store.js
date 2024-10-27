import { configureStore } from '@reduxjs/toolkit';
import personReducer from './reducers/nameReducer.js';

export const store = configureStore({
  reducer: {
    person: personReducer,
  },
});

export default store;

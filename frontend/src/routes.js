import React from 'react';

import { Icon } from '@chakra-ui/react';
import {
  MdBarChart,
  MdPerson,
  MdHome,
  MdLock,
  MdOutlineShoppingCart,
  MdAccountTree,
  MdPolyline,
  MdAddTask,
  MdListAlt,
  MdScheduleSend
} from 'react-icons/md';


import GraphIcon from "./images/graph.svg"

// Admin Imports
import MainDashboard from 'views/admin/default';
import NFTMarketplace from 'views/admin/marketplace';
// import Profile from 'views/admin/profile';
import DataTables from 'views/tasks';
import Validate from 'views/validate';
import Profile  from 'views/profile';
import Training from 'views/TrainingPage';
import RTL from 'views/admin/rtl';

// Auth Imports
import SignInCentered from 'views/auth/signIn';

const routes = [
  /*{
    name: 'Main Dashboard',
    layout: '/admin',
    path: '/default',
    icon: <Icon as={MdHome} width="20px" height="20px" color="inherit" />,
    component: <MainDashboard />,
  },*/

  /*{
    name: 'NFT Marketplace',
    layout: '/admin',
    path: '/nft-marketplace',
    icon: (
      <Icon
        as={MdOutlineShoppingCart}
        width="20px"
        height="20px"
        color="inherit"
      />
    ),
    component: <NFTMarketplace />,
    secondary: true,
  },*/
  
  {
    name: 'Task Creation',
    layout: '/admin',
    icon: <Icon as={MdAddTask} width="20px" height="20px" color="inherit" />,
    path: '/task-creation',
    component: <DataTables />,
  },
  {
    name: 'Training Node',
    layout: '/admin',
    icon: <Icon as={MdPolyline} width="20px" height="20px" color="inherit" />,
    path: '/data-tables1',
    component: <Training />,
  },
  {
    name: 'Validate',
    layout: '/admin',
    icon: <Icon as={MdListAlt} width="20px" height="20px" color="inherit" />,
    path: '/data-tables2',
    component: <Validate />,
  },
  {
    name: 'Delegate',
    layout: '/admin',
    icon: <Icon as={MdScheduleSend} width="20px" height="20px" color="inherit" />,
    path: '/data-tables3',
    component: <DataTables />,
  },
  {
    name: 'Profile',
    layout: '/admin',
    path: '/profile',
    icon: <Icon as={MdPerson} width="20px" height="20px" color="inherit" />,
    component: <Profile />,
  },
  {
    name: 'Sign In',
    layout: '/sign-in',
    path: '/',
    icon: <Icon as={MdLock} width="20px" height="20px" color="inherit" />,
    component: <SignInCentered />,
  },
];

export default routes;

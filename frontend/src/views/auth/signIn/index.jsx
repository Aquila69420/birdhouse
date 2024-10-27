/* eslint-disable */

import React, { useState } from "react";
import axios from "axios";
import { Link, useNavigate } from "react-router-dom";
import { useSelector, useDispatch } from "react-redux";
import { setName, setAddress, setIsLoggedIn} from "../../../redux/reducers/nameReducer";
import {
  Box,
  Button,
  Flex,
  FormControl,
  FormLabel,
  Heading,
  Input,
  Text,
  useColorModeValue,
} from "@chakra-ui/react";

function SignIn() {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  
  // Local state to manage form inputs
  const [name, setNameInput] = useState('');
  const [address, setAddressInput] = useState('');

  // Fetching stored values to check or display if needed
  const storedName = useSelector((state) => state.person.name);
  const storedAddress = useSelector((state) => state.person.address);

  const handleSignIn = () => {
    // Dispatching to update Redux state
    dispatch(setName(name));
    dispatch(setAddress(address));
    dispatch(setIsLoggedIn(true))
    let login_successful = false;
    // TODO: Check the db if the user already exists then sign in else register
    axios.post("http://10.154.36.81:5000/login_client", {
      wallet_address: address,
    }).then((res) => {
      if (res.status == 404) {
        axios.post("http://10.154.36.81:5000/login_client", {
          wallet_address: address,
        })
      }
    })

    // Navigate to the admin page after signing in
    navigate("/admin/task-creation");
  };

  const textColor = useColorModeValue("navy.700", "white");
  const textColorSecondary = "gray.400";
  const textColorBrand = useColorModeValue("brand.500", "white");

  return (
    <Flex
      minH="100vh"
      align="center"
      justify="center"
      bg="gray.50"
      p="24px"
    >
      <Box
        w="100%"
        maxW="420px"
        bg="white"
        boxShadow="lg"
        p="24px"
        borderRadius="8px"
      >
        <Heading color={textColor} fontSize="2xl" mb="10px" textAlign="center">
          Sign In
        </Heading>
        <Text mb="36px" color={textColorSecondary} textAlign="center">
          Enter your name and wallet address to sign in!
        </Text>
        
        <FormControl mb="24px" isRequired>
          <FormLabel color={textColor}>Name</FormLabel>
          <Input
            type="text"
            placeholder="Your Name"
            value={name}
            onChange={(e) => setNameInput(e.target.value)}
          />
        </FormControl>
        
        <FormControl mb="24px" isRequired>
          <FormLabel color={textColor}>Wallet Address</FormLabel>
          <Input
            type="text"
            placeholder="Wallet Address"
            value={address}
            onChange={(e) => setAddressInput(e.target.value)}
          />
        </FormControl>
        <Link to="/admin">
        <Button
          fontSize="sm"
          fontWeight="500"
          w="100%"
          h="50px"
          mb="12px"
          backgroundColor="#4318ff"
          color="white"
          _hover={{ backgroundColor: "#3a16e0" }}  // Darker shade for hover effect
          _active={{ backgroundColor: "#3214c2" }}  // Even darker for active state
          onClick={handleSignIn}
        >
          Sign In
        </Button>
        </Link>
      </Box>
    </Flex>
  );
}

export default SignIn;

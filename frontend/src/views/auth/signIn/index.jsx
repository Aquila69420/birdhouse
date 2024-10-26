/* eslint-disable */

import React from "react";
import { NavLink } from "react-router-dom";
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
          <Input type="text" placeholder="Your Name" />
        </FormControl>
        <FormControl mb="24px" isRequired>
          <FormLabel color={textColor}>Wallet Address</FormLabel>
          <Input type="text" placeholder="Wallet Address" />
        </FormControl>
        <Button colorScheme="teal" w="100%" mb="12px">
          Sign In
        </Button>
      </Box>
    </Flex>
  );
}

export default SignIn;

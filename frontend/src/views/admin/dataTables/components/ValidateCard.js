// ValidateCard.js

import React, { useState } from "react";
import {
  Box,
  Button,
  FormControl,
  FormLabel,
  Input,
  Select,
  useColorModeValue,
} from "@chakra-ui/react";
import Card from "components/card/Card"; // Import your Card component for consistent styling

const ValidateCard = ({ tasks, onStake }) => {
  const [selectedTask, setSelectedTask] = useState("");
  const [tokens, setTokens] = useState("");

  // Colors and styles from useColorModeValue called only once at the top
  const textColor = useColorModeValue("secondaryGray.900", "white");
  const buttonBgColor = useColorModeValue("brand.500", "brand.400");
  const buttonHoverColor = useColorModeValue("brand.600", "brand.300");
  const buttonActiveColor = useColorModeValue("brand.700", "brand.200");

  // Handle task selection and show/hide the token input field
  const handleTaskChange = (e) => {
    setSelectedTask(e.target.value);
    setTokens(""); // Reset tokens when a new task is selected
  };

  const handleStake = () => {
    onStake({ task: selectedTask, tokens });
    setSelectedTask("");
    setTokens("");
  };

  return (
    <Card
      p="20px"
      borderRadius="20px"
      boxShadow="lg"
      w="100%"
      h="100%"
      bg={useColorModeValue("white", "navy.900")}
    >
      <FormControl mb="4">
        <FormLabel fontWeight="600" color={textColor}>
          Select Task
        </FormLabel>
        <Select
          placeholder="Choose a task to validate"
          value={selectedTask}
          onChange={handleTaskChange}
          variant="outline"
          fontSize="sm"
        >
          {tasks.map((task) => (
            <option key={task.id} value={task.name}>
              {task.name}
            </option>
          ))}
        </Select>
      </FormControl>

      {selectedTask && (
        <>
          <FormControl mb="4">
            <FormLabel fontWeight="600" color={textColor}>
              Tokens to Stake
            </FormLabel>
            <Input
              type="number"
              value={tokens}
              onChange={(e) => setTokens(e.target.value)}
              placeholder="Enter amount"
              variant="outline"
              fontSize="sm"
            />
          </FormControl>
          <Button
            colorScheme="brandScheme"
            bg={buttonBgColor}
            color="white"
            fontSize="sm"
            fontWeight="500"
            w="100%"
            h="50px"
            borderRadius="12px"
            onClick={handleStake}
            _hover={{ bg: buttonHoverColor }}
            _active={{ bg: buttonActiveColor }}
          >
            Stake FML
          </Button>
        </>
      )}
    </Card>
  );
};

export default ValidateCard;

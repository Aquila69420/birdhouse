// TaskCreationCard.js

import React, { useState } from "react";
import {
  Box,
  Button,
  FormControl,
  FormLabel,
  Input,
  Select,
  Textarea,
  useColorModeValue,
} from "@chakra-ui/react";
import Card from "components/card/Card"; // Import your Card component

const TaskCreationCard = ({ onCreateTask }) => {
  const [model, setModel] = useState("LLM"); // Default model set to LLM
  const [name, setName] = useState(""); // New name field state
  const [parameters, setParameters] = useState({
    layers: "",
    tokens: "",
    temperature: ""
  });
  const [description, setDescription] = useState("");

  // Model-specific parameters
  const modelParameters = {
    CNN: { layers: "", learningRate: "" },
    LR: { learningRate: "", regularization: "" },
    DT: { maxDepth: "", minSamplesSplit: "" },
    LLM: { layers: "", tokens: "", temperature: "" },
  };

  // Handle model selection
  const handleModelChange = (event) => {
    const selectedModel = event.target.value;
    setModel(selectedModel);
    setParameters(modelParameters[selectedModel]);
  };

  // Handle parameter change
  const handleParameterChange = (param, value) => {
    setParameters((prevParams) => ({
      ...prevParams,
      [param]: value,
    }));
  };

  const handleCreateTask = () => {
    const taskData = { model, name, parameters, description }; // Include name in taskData
    onCreateTask(taskData);
    // Reset fields
    setModel("LLM"); // Reset to default LLM
    setParameters(modelParameters["LLM"]);
    setName("");
    setDescription("");
  };

  // Theme colors for consistent styling
  const textColor = useColorModeValue("secondaryGray.900", "white");

  return (
    <Card
      p="20px"
      borderRadius="20px" // Adjust to match other cards
      boxShadow="lg"
      w="100%"
      h="100%"
      bg={useColorModeValue("white", "navy.900")}
    >
      <FormControl mb="4">
        <FormLabel fontWeight="600" color={textColor}>
          Task Name
        </FormLabel>
        <Input
          variant="outline"
          fontSize="sm"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Enter task name"
        />
      </FormControl>

      <FormControl mb="4">
        <FormLabel fontWeight="600" color={textColor}>
          Model
        </FormLabel>
        <Select
          variant="outline"
          fontSize="sm"
          value={model}
          onChange={handleModelChange}
        >
          <option value="CNN">CNN</option>
          <option value="LR">Logistic Regression</option>
          <option value="DT">Decision Tree</option>
          <option value="LLM">Large Language Model</option>
        </Select>
      </FormControl>

      {Object.keys(parameters).map((param) => (
        <FormControl key={param} mb="4">
          <FormLabel fontWeight="600" color={textColor}>
            {param.charAt(0).toUpperCase() + param.slice(1)}
          </FormLabel>
          <Input
            variant="outline"
            fontSize="sm"
            value={parameters[param]}
            onChange={(e) => handleParameterChange(param, e.target.value)}
            placeholder={`Enter ${param}`}
          />
        </FormControl>
      ))}

      <FormControl mb="4">
        <FormLabel fontWeight="600" color={textColor}>
          Description
        </FormLabel>
        <Textarea
          variant="outline"
          fontSize="sm"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Task description"
        />
      </FormControl>
      <Button
        colorScheme="brandScheme" // Use your theme's brand color scheme
        bg={useColorModeValue("brand.500", "brand.400")} // Match button color with the app's primary color
        color="white"
        fontSize="sm"
        fontWeight="500"
        w="100%"
        h="50px"
        borderRadius="12px" // Match with button styling in other components
        _hover={{ bg: useColorModeValue("brand.600", "brand.300") }}
        _active={{ bg: useColorModeValue("brand.700", "brand.200") }}
        onClick={handleCreateTask}
      >
        Create Task
      </Button>
    </Card>
  );
};

export default TaskCreationCard;

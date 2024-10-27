// Chakra imports
import { Box, SimpleGrid } from "@chakra-ui/react";
import ComplexTable from "views/admin/dataTables/components/ComplexTable";
import TaskCreationCard from "views/admin/dataTables/components/TaskCreationCard"; // Import the TaskCreationCard
import { columnsDataComplex } from "views/admin/dataTables/variables/columnsData";
import tableDataComplex from "views/admin/dataTables/variables/tableDataComplex.json";
import React from "react";
import axios from "axios";

export default function Settings() {
  // Handle task creation (e.g., add to table or call an API)
  const handleCreateTask = (taskData) => {
    console.log("New Task Created:", taskData);
    switch(taskData.model) {
      case "LLM":
        axios.post('http://10.154.36.81:5000/instantiate_llm_flock_model', {
          model_name: 'gpt2'
        })
      break;
      case "CNN":
        axios.post('http://10.154.36.81:5000/instantiate_flock_model', {
          model_name: 'CNN',
          loss_function: 'CE'
        })
        break;
      case "CNN":
        axios.post('http://10.154.36.81:5000/instantiate_flock_model', {
          model_name: 'LR',
          loss_function: 'BCE'
        })
        break;
      case "DT":
        axios.post('http://10.154.36.81:5000/instantiate_flock_model', {
            model_name: 'DT',
            loss_function: 'MSE'
        })
        break;
    }
  };

  return (
    <Box
      pt={{ base: "130px", md: "80px", xl: "80px" }}
      h="100vh" // Full viewport height
      overflow="auto" // Enable scrolling if table content exceeds viewport
    >
      <SimpleGrid
        gridTemplateColumns={{ sm: "1fr", md: "30% 70%" }} // 30% for TaskCreationCard, 70% for ComplexTable on larger screens
        spacing="20px"
        h="100%"
      >
        {/* Task creation card */}
        <TaskCreationCard onCreateTask={handleCreateTask} />
        
        {/* Table component */}
        <ComplexTable
          columnsData={columnsDataComplex}
          tableData={tableDataComplex}
        />
      </SimpleGrid>
    </Box>
  );
}

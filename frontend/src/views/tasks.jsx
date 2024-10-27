// Chakra imports
import { Box, SimpleGrid } from "@chakra-ui/react";
import ComplexTable from "views/admin/dataTables/components/ComplexTable";
import TaskCreationCard from "views/admin/dataTables/components/TaskCreationCard"; // Import the TaskCreationCard
import { columnsDataComplex } from "views/admin/dataTables/variables/columnsData";
import tableDataComplexJson from "views/admin/dataTables/variables/tableDataComplex.json"; // Import JSON as initial data
import React, { useState } from "react";
import axios from "axios";

export default function Settings() {
  // Use state to manage table data so it can be updated dynamically
  const [tableDataComplex, setTableDataComplex] = useState(tableDataComplexJson);

  // Handle task creation
  const handleCreateTask = async (taskData) => {
    console.log("New Task Created:", taskData);

    // API request depending on the selected model
    try {
      switch (taskData.model) {
        case "LLM":
          await axios.post('http://10.154.36.81:5000/instantiate_llm_flock_model', {
            model_name: 'gpt2'
          });
          break;
        case "CNN":
          await axios.post('http://10.154.36.81:5000/instantiate_flock_model', {
            model_name: 'CNN',
            loss_function: 'CE'
          });
          break;
        case "LR":
          await axios.post('http://10.154.36.81:5000/instantiate_flock_model', {
            model_name: 'LR',
            loss_function: 'BCE'
          });
          break;
        case "DT":
          await axios.post('http://10.154.36.81:5000/instantiate_flock_model', {
            model_name: 'DT',
            loss_function: 'MSE'
          });
          break;
        default:
          console.warn("Unknown model type");
          return;
      }

      // Find the maximum task-id in the current data and increment it by 1
      const newTaskId = (Math.max(...tableDataComplex.map(task => parseInt(task["task-id"])), 0) + 1).toString();

      // Format the new task with the same structure
      const newTask = {
        "task-id": newTaskId,
        "task-name": taskData.model,
        "task-status": "Submission",
        "date": new Date().toLocaleDateString("en-US", { year: 'numeric', month: 'short', day: 'numeric' }),
        "progress": 0,
        "percentage": 0,
        "description": taskData.description || ""
      };
      
      // Update table data with the new task
      setTableDataComplex((prevData) => [...prevData, newTask]);

    } catch (error) {
      console.error("Error creating task:", error);
      alert("Failed to create task. Please try again.");
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
        
        {/* Table component with updated tableData */}
        <ComplexTable
          columnsData={columnsDataComplex}
          tableData={tableDataComplex}
        />
      </SimpleGrid>
    </Box>
  );
}

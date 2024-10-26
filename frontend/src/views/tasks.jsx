// Chakra imports
import { Box, SimpleGrid } from "@chakra-ui/react";
import ComplexTable from "views/admin/dataTables/components/ComplexTable";
import TaskCreationCard from "views/admin/dataTables/components/TaskCreationCard"; // Import the TaskCreationCard
import { columnsDataComplex } from "views/admin/dataTables/variables/columnsData";
import tableDataComplex from "views/admin/dataTables/variables/tableDataComplex.json";
import React from "react";

export default function Settings() {
  // Handle task creation (e.g., add to table or call an API)
  const handleCreateTask = (taskData) => {
    console.log("New Task Created:", taskData);
    // Add logic to save or process the new task
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

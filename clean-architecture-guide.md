# Clean Architecture Guide

This guide outlines a refined approach to structuring your project using Clean Architecture principles. The goal is to achieve a modular, scalable, and testable codebase where business logic remains independent from external systems such as databases, APIs, or frameworks.

---

## Overview

Clean Architecture divides the system into several layers, each with a clear responsibility. The key principles are:

- **Independence of Frameworks:**  
  Inner layers (Domain and Application) remain independent of external technologies (UI, database, third-party services).

- **Dependency Inversion:**  
  All dependencies point inward; core business logic does not depend on outer layers.

- **Interface Adapters:**  
  Adapters convert data between external formats and internal representations.

- **Incremental Migration:**  
  Migrate features gradually so the system remains functional during the transition.

---

## Structure

. ├── domain/ # Enterprise business rules
  │ ├── entities/ # Core business objects (e.g., User, Order)
  │ ├── value_objects/ # Immutable objects representing domain concepts 
  │ └── interfaces/ # Abstract services and repository interfaces 
│ ├── application/ # Application business rules 
  │ ├── dtos/ # Data Transfer Objects for input/output
  │ ├── use_cases/ # Application services or interactors 
  │ ├── interfaces/ # Port interfaces for services if needed 
  │ └── exceptions/ # Application-specific exceptions 
│ ├── infrastructure/ # Tools and external services 
  │ ├── persistence/ # Database implementations and Vector DBs 
  │ │ ├── repositories/ # Concrete repository implementations 
  │ │ └── models/ # ORM models or database schemas 
  │ ├── external/ # External service integrations (e.g., LLM, API clients) 
  │ └── config/ # Service-specific configuration and settings 
│ ├── presentation/ # User interface layer 
  │ ├── api/ # HTTP routes, controllers, endpoints (e.g., FastAPI) 
  │ │ ├── v1/ # Versioned API endpoints 
  │ │ └── middlewares/ # Request/response handlers and exception handlers 
  │ └── cli/ # (Optional) Command-line interfaces 
│ └── common/ # Shared functionality 
├── utils/ # General utility functions and helpers 
└── constants/ # Shared constants and configuration defaults


## Layer Descriptions

### 1. Domain Layer

**Purpose:**  
Encapsulate the core business logic without any external dependencies.

**Components:**

- **Entities:**  
  Pure business objects representing your core models (e.g., User, Order). They encapsulate business rules and are independent of external systems.

- **Value Objects:**  
  Represent immutable domain concepts (e.g., Email, Money) with validation and business rules.

- **Interfaces:**  
  Define contracts for repositories and other external services required by the domain. These abstractions ensure that the core business logic is not coupled with technical details.

---

### 2. Application Layer

**Purpose:**  
Implement application-specific business rules by orchestrating interactions between domain entities and external dependencies.

**Components:**

- **DTOs (Data Transfer Objects):**  
  Specify the input and output data formats for use cases. DTOs act as a bridge between external input (e.g., HTTP requests) and internal domain models.

- **Use Cases / Interactors:**  
  Contain the application-specific business rules. They coordinate processes such as user registration, login, and token validation by calling repository methods via defined interfaces.

- **Interfaces:**  
  (Optionally) Additional service interfaces at this level to abstract specific application services.

- **Exceptions:**  
  Custom exceptions for handling application-specific error scenarios, ensuring consistent error handling across use cases.

---

### 3. Infrastructure Layer

**Purpose:**  
Handle external dependencies like databases, third-party APIs, integrations, and configuration management.

**Components:**

- **Persistence:**  
  Contains database connection logic, ORM models, and repository implementations. The persistence layer encapsulates the technical details of data access, ensuring that changes to the database or data source do not affect the domain or application layers.

- **External Services:**  
  Houses integrations with third-party services (e.g., LLM integrations, external API clients). This layer isolates external dependencies from the core business logic.

- **Configuration:**  
  Manages settings for databases, external APIs, and other technical dependencies. Service-specific configurations belong here, keeping the application-level configuration separate.

---

### 4. Presentation Layer

**Purpose:**  
Handle HTTP requests/responses, user interactions, and dependency injection into the Application Layer.

**Components:**

- **API Endpoints / Controllers:**  
  Expose the application to the outside world by handling incoming requests, validating input via DTOs, invoking use cases, and formatting responses.

- **Middlewares & Exception Handlers:**  
  Provide global error handling and request/response processing to ensure consistency across the application.

- **Dependency Injection:**  
  Use mechanisms (e.g., FastAPI's `Depends`) to inject use cases and services into route handlers, enabling loose coupling and ease of testing.

---

### 5. Common Layer

**Purpose:**  
Store shared utilities, constants, and helper functions that do not belong to any specific layer.

**Components:**

- **Utilities:**  
  Contains helper functions and shared code used across multiple layers.

- **Constants:**  
  Holds configuration defaults and shared constants.

---

## Service-by-Service Division

For each service (e.g., Authentication, LLM Integration), responsibilities are divided as follows:

- **Domain:**  
  Define the core business entities (e.g., User) and repository interfaces (e.g., IAuthRepository) for that service.

- **Application:**  
  Create DTOs for input/output, implement use cases (e.g., AuthUseCase) that encapsulate service-specific business logic (such as registration, login, and token management), and handle custom exceptions.

- **Infrastructure:**  
  Implement concrete repository classes (e.g., JSON-based or SQL-based repositories) and external service integrations (e.g., LLM service, external API clients). Also, include service-specific configuration (e.g., database connection details, API keys).

- **Presentation:**  
  Build API endpoints (e.g., `/api/v1/auth/login`) that invoke use cases, validate incoming data, and return responses. Set up global exception handlers and middleware as needed.

---

## Key Principles

- **Independence of Frameworks:**  
  Inner layers (Domain and Application) remain independent of external technologies (UI, database, third-party services). This means that changes in frameworks or external systems should not affect the core business logic.

- **Dependency Inversion:**  
  All dependencies point inward. The Domain layer does not depend on the Application, Infrastructure, or Presentation layers. Instead, outer layers depend on abstractions defined in the inner layers.

- **Interface Adapters:**  
  Use adapters to convert data from external formats (like API requests or ORM models) into internal domain objects and vice versa. This maintains a clear separation between how data is received/stored and how it is used in business logic.

- **Incremental Migration:**  
  Refactor and migrate features gradually. Maintain a working system while moving individual features to the new structure, ensuring smooth transitions without disrupting current functionality.

- **Service-Specific Isolation:**  
  Each service (e.g., Authentication, LLM Integration) is divided across layers so that its domain, application, infrastructure, and presentation concerns are all independently manageable and testable.

---

## Final Thoughts

By adopting this Clean Architecture structure, you achieve:

- **Modularity:**  
  Each layer is isolated, which makes the codebase easier to maintain, test, and extend.

- **Testability:**  
  Business logic can be tested in isolation from external dependencies.

- **Flexibility:**  
  External systems (databases, APIs) can be swapped without affecting core business logic.

- **Scalability:**  
  Services can be refactored independently as the system grows.



# MOSAIC Java

This is the Java implementation of the MOSAIC project - Model Orchestration & Synthesis for Adaptive Intelligent Computation.

## Building

To build the project, use Maven:

```bash
mvn clean compile
```

## Running

To run the application:

```bash
mvn exec:java -Dexec.mainClass="com.jiva.mosaic.Main"
```

## Testing

To run tests:

```bash
mvn test
```

## Project Structure

```
mosaic-java/
├── pom.xml
├── src/
│   ├── main/
│   │   ├── java/
│   │   │   └── com/jiva/mosaic/
│   │   └── resources/
│   └── test/
│       ├── java/
│       │   └── com/jiva/mosaic/
│       └── resources/
└── README.md
```


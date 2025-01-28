import streamlit as st

def main():
  st.title("Clasificación de la base de datos MNIST")
  st.markdown("Sube una imagen para clasificar")

  uploaded_file=st.file.uploader("Selecciona una imagen PNG, JPG JEPG: "), type = ["jpg", "png", "jpeg"])

if __name__ == "__main__":
  main()

# Kavak LLM agent

## Este agente soporta las siguientes caracteristicas:
- Apalanca uso de RAG para injeccion de contexto sobre catalogo
- Guarda memoria de conversaciones previas
- Ofrece y personaliza experiencia de usuarios ofreciendo ofertas del catalogo `sample_caso_ai_engineer.csv`
- Interactua por Whatsapp a traves de Twilio

## Setup rapido

1. Obtener key de OpenAI y guardar en .env (Copiar .env.template a .env en raiz)
2. Correr `make run`
3. Correr `make ngrok` y copiar URL generada (Finaliza con .ngrok-free.app)
4. Ir [Twilio Whatsapp Sandbox ](https://console.twilio.com/us1/develop/sms/try-it-out/whatsapp-learn?_gl=1*1ohbgtn*_ga*MTQwMjUxMDAyLjE3NjIzMTUwOTc.*_ga_RRP8K4M4F3*czE3NjIzMTU5MzMkbzEkZzEkdDE3NjIzMTcyODUkajQ1JGwwJGgw) y luego a *Sandbox Settings*
5. Remplazar campo **When a message comes in** con `https://<REMPLAZA>.ngrok-free.app/ask`
6. Dar de alta numero a Whatsapp Sandbox
7. Listo! Ya puedes probar este agente desde tu cuenta de Whatsapp!


## Diagrama de componentes

## Diagrama de pipeline de agente LLM

## Roadmap propuesto
